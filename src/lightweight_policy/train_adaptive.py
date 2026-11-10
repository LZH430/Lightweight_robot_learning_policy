"""Independent experimental runner. Legacy train.py remains unchanged."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from collections import Counter
from .model_adaptive import AdaptiveStudentPolicy as StudentPolicy
from .train import FeatureCache, balanced_sampler
from scripts.losses_adaptive import loss_and_metrics, validate_data_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--init-from", help="Weights only; starts a new stage and optimizer")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("implementation") != "adaptive_v1":
        raise ValueError("Use a new adaptive_v1 config with this runner")
    validate_data_config(cfg)
    torch.manual_seed(cfg.get("seed",42))
    stage = cfg.get("stage", "joint")
    if stage == "action" and not (args.init_from or args.resume):
        raise ValueError("Frozen-future action training requires --init-from or --resume")
    if cfg.get("teacher_weight",0) < 0 or cfg.get("future_weight",.1) < 0:
        raise ValueError("Loss weights must be nonnegative")
    device = torch.device(cfg.get("device","cuda"))
    model = StudentPolicy(**cfg["model"]).to(device)
    model.set_stage(stage)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best = 0, float("inf")
    if args.resume or args.init_from:
        saved = torch.load(args.resume or args.init_from, map_location="cpu", weights_only=True)
        if saved["config"].get("implementation") != "adaptive_v1":
            raise ValueError("Legacy checkpoints are not compatible; train this experiment from scratch")
        if saved["config"]["model"] != cfg["model"]:
            raise ValueError("Checkpoint model configuration mismatch")
        model.load_state_dict(saved["model"], strict=True)
        if args.resume:
            for key in ("adaptive_distillation", "adaptive_temperature", "teacher_weight", "future_weight",
                        "adaptive_data", "data_weight", "data_temperature"):
                if saved["config"].get(key) != cfg.get(key):
                    raise ValueError(f"Resume setting changed: {key}; use --init-from for a new experiment")
            if saved["config"].get("stage","joint") != stage:
                raise ValueError("Use --init-from to change training stage")
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            start, best = saved["epoch"]+1, saved.get("best", float("inf"))
    train_data = FeatureCache(cfg["train_cache"],cfg,True)
    sampler = balanced_sampler(train_data,cfg["dataset_weights"]) if "dataset_weights" in cfg else None
    train = DataLoader(train_data,batch_size=cfg["batch_size"],sampler=sampler,shuffle=sampler is None)
    val = DataLoader(FeatureCache(cfg["val_cache"],cfg), batch_size=cfg["batch_size"])
    if {p.resolve() for p in train.dataset.files} & {p.resolve() for p in val.dataset.files}:
        raise ValueError("Train and validation cache files overlap")
    output = Path(cfg["output"])
    output.mkdir(parents=True,exist_ok=True)
    if not args.resume and any((output / name).exists() for name in
                               ("last.ckpt", "best.ckpt", "last.pt", "best.pt")):
        raise ValueError("Output already contains a checkpoint; resume or choose a new directory")
    print(f"stage={stage} trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    for epoch in range(start,cfg["epochs"]):
        if sampler is not None:
            sampler.generator.manual_seed(cfg.get("seed",42)+epoch)
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
        for training, loader in ((True,train),(False,val)):
            model.train(training)
            totals, counts = {}, {}
            dataset_totals, dataset_counts = {}, {}
            sampled_counts = Counter()
            for sample in loader:
                b = {k:v.to(device) for k,v in sample.items() if isinstance(v,torch.Tensor)}
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.autocast(device.type,enabled=amp,dtype=torch.float16):
                    result = model(b["visual"],b["text"],b["state"])
                    loss, metrics = loss_and_metrics(model,b,cfg,training,result)
                names = sample.get("dataset_name",sample.get("schema_dataset"))
                if names is None and cfg.get("dataset_weights"):
                    raise ValueError("Dataset identity required for per-dataset evaluation")
                if names is not None:
                    sampled_counts.update(names)
                    if not training:
                        for name in sorted(set(names)):
                            indices = torch.tensor([i for i,n in enumerate(names) if n==name],device=device)
                            group = {k:v.index_select(0,indices) for k,v in b.items()}
                            predictions = {k:v.index_select(0,indices) for k,v in result.items()}
                            _, group_metrics = loss_and_metrics(model,group,cfg,False,predictions)
                            for key,(value,count) in group_metrics.items():
                                metric_key = (name,key)
                                dataset_totals[metric_key] = dataset_totals.get(metric_key,0.)+value*count
                                dataset_counts[metric_key] = dataset_counts.get(metric_key,0)+count
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training/validation loss")
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                    scaler.step(optimizer)
                    scaler.update()
                for key,(value,count) in metrics.items():
                    totals[key] = totals.get(key,0.) + value*count
                    counts[key] = counts.get(key,0) + count
            values = {k:totals[k]/counts[k] for k in totals}
            row = dict(epoch=epoch,split="train" if training else "val",stage=stage,**values)
            row["dataset_sample_counts"] = dict(sampled_counts)
            if not training and dataset_totals:
                groups = {}
                for (name,key),value in dataset_totals.items():
                    groups.setdefault(name,{})[key] = value/dataset_counts[(name,key)]
                expected = set(cfg.get("dataset_weights",groups))
                if set(groups) != expected:
                    raise ValueError("Validation datasets do not match configured training datasets")
                if len(groups) > 1:
                    row["by_dataset"] = groups
                    row["macro"] = {key:sum(group[key] for group in groups.values())/len(groups)
                                    for key in next(iter(groups.values()))}
            print(json.dumps(row),flush=True)
            with (output/"metrics.jsonl").open("a",encoding="utf-8") as file:
                file.write(json.dumps(row)+"\n")
        score = values["future_mse" if stage == "future" else "action_mse"]
        improved = score < best
        best = min(best,score)
        saved = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                     epoch=epoch,config=cfg,best=best)
        torch.save(saved,output/"last.ckpt")
        if improved:
            torch.save(saved,output/"best.ckpt")
        if amp:
            print(f"peak_allocated_gib={torch.cuda.max_memory_allocated(device)/2**30:.3f}")


if __name__ == "__main__":
    main()
