import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import math
from .model import StudentPolicy, masked_mse


class FeatureCache(Dataset):
    def __init__(self, directory, cfg=None, training=False):
        self.files = sorted(Path(directory).glob("*.pt"))
        self.cfg, self.training = cfg, training
        if not self.files:
            raise ValueError(f"No cached samples found in {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        b = torch.load(path, map_location="cpu", weights_only=True)
        if self.cfg:
            c = self.cfg["model"]
            shapes = {"visual": (c.get("obs_steps", 2), c.get("tokens", 64), c["visual_dim"]),
                      "text": (c["text_dim"],), "state": (c.get("state_dim",16),),
                      "action": (c.get("horizon",16), c.get("action_dim",16)),
                      "action_mask": (c.get("horizon",16), c.get("action_dim",16))}
            if c.get("future", False):
                shapes["future"] = (c.get("tokens",64), c["visual_dim"])
            if self.training and self.cfg.get("teacher_weight",0) and self.cfg.get("stage","joint") != "future":
                shapes["teacher_action"] = shapes["action"]
            for key, shape in shapes.items():
                if key not in b or not isinstance(b[key], torch.Tensor) or tuple(b[key].shape) != shape:
                    raise ValueError(f"{path}: {key} must be a tensor with shape {shape}")
                if not b[key].is_floating_point():
                    b[key] = b[key].float()
            if self.cfg.get("implementation") in ("dynamics_v1", "dynamics_v2"):
                required = {"teacher_future": (c.get("tokens",64), c["visual_dim"]),
                            "dynamics_action": shapes["action"]}
                if self.cfg.get("dynamics_real_weight", 0) or self.cfg.get("adaptive_dynamics", False):
                    required["future"] = required["teacher_future"]
                for key, shape in required.items():
                    value = b.get(key)
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                        raise ValueError(f"{path}: missing/invalid dynamics tensor {key}, expected {shape}")
                    b[key] = value.float()
                if b.get("dynamics_fs") != self.cfg.get("dynamics_fs") or b.get("dynamics_fs_type") != self.cfg.get("dynamics_fs_type"):
                    raise ValueError(f"{path}: dynamics sampling rate does not match training config")
                if b.get("dynamics_target_frame") != c.get("horizon",16)-1:
                    raise ValueError(f"{path}: dynamics target frame mismatch")
            mask = b["action_mask"]
            if not ((mask == 0) | (mask == 1)).all() or not mask.any():
                raise ValueError(f"{path}: action_mask must contain valid binary entries")
            for key in shapes:
                values = b[key][mask.bool()] if key in ("action", "teacher_action") else b[key]
                if not torch.isfinite(values).all():
                    raise ValueError(f"{path}: non-finite values in {key}")
        if self.cfg and self.cfg.get("implementation") == "action_only_v1":
            inputs = {"visual", "text", "state", "action", "action_mask"}
            b = {k:v for k,v in b.items() if k in inputs or not isinstance(v, torch.Tensor)}
        return b


def balanced_sampler(dataset, weights):
    if not isinstance(weights, dict) or not weights:
        raise ValueError("dataset_weights must be a nonempty mapping")
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("Dataset weights must be finite and positive")
    names = []
    for path in dataset.files:
        sample = torch.load(path,map_location="cpu",weights_only=True)
        name = sample.get("dataset_name",sample.get("schema_dataset"))
        if not isinstance(name,str) or name not in weights:
            raise ValueError(f"{path}: missing or unconfigured dataset_name: {name}")
        names.append(name)
    counts = Counter(names)
    if set(counts) != set(weights):
        raise ValueError(f"No training samples for datasets: {set(weights)-set(counts)}")
    # Each dataset's total probability mass equals its configured weight.
    sample_weights = [weights[name]/counts[name] for name in names]
    print("Training dataset counts:",dict(counts),"target probabilities:",
          {name:value/sum(weights.values()) for name,value in weights.items()})
    return WeightedRandomSampler(sample_weights,len(dataset),replacement=True,
                                 generator=torch.Generator())


def loss_and_metrics(model, b, cfg, training, result=None):
    if result is None:
        result = model(b["visual"], b["text"], b["state"])
    terms, metrics = [], {}
    if "action" in result:
        loss = masked_mse(result["action"], b["action"], b["action_mask"])
        terms.append(loss)
        metrics["action_mse"] = (loss.detach().item(), int(b["action_mask"].bool().sum()))
        if training and cfg.get("teacher_weight", 0):
            kd = masked_mse(result["action"], b["teacher_action"], b["action_mask"])
            terms.append(cfg["teacher_weight"] * kd)
            metrics["teacher_mse"] = (kd.detach().item(), int(b["action_mask"].bool().sum()))
    if model.future:
        predicted, target = result["future"].float(), b["future"].float()
        mse = (predicted - target).square().mean()
        copy_mse = (b["visual"][:, -1].float() - target).square().mean()
        cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
        metrics.update(future_mse=(mse.detach().item(), target.numel()),
                       future_copy_mse=(copy_mse.item(), target.numel()),
                       future_cosine=(cosine.detach().item(), target.shape[0]*target.shape[1]))
        if model.stage != "action":
            terms.append((1.0 if model.stage == "future" else cfg.get("future_weight", .1)) * mse)
    return sum(terms), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--init-from", help="Weights only; starts a new stage and optimizer")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("implementation") in ("dynamics_v1", "dynamics_v2"):
        raise ValueError("Dynamic distillation configs require python -m lightweight_policy.train_dynamics")
    if cfg.get("implementation") == "action_only_v1" and (
        cfg["model"].get("future", False) or cfg.get("teacher_weight", 0)
    ):
        raise ValueError("action_only_v1 requires future=false and teacher_weight=0")
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
        if saved["config"]["model"] != cfg["model"]:
            raise ValueError("Checkpoint model configuration mismatch")
        model.load_state_dict(saved["model"], strict=True)
        if args.resume:
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
