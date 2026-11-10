"""Independent experimental runner. Legacy train.py remains unchanged."""
import argparse
import math
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from .model_dynamics import DynamicsStudentPolicy as StudentPolicy
from .train import FeatureCache, balanced_sampler
from scripts.losses_dynamics import loss_and_metrics, validate_data_config
from .gripper_finetune import checkpoint_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--init-from", help="Weights only; overrides JSON init_from and starts a new optimizer")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # CLI checkpoint options take precedence over the JSON initialization source.
    if args.resume is None and args.init_from is None:
        args.init_from = cfg.get("init_from")
    if args.init_from is not None:
        if not isinstance(args.init_from, str) or not args.init_from.strip():
            parser.error("init_from must be a nonempty checkpoint path or null")
        args.init_from = str(Path(args.init_from).expanduser())
        if not Path(args.init_from).is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_from}")
    if cfg.get("implementation") != "dynamics_v2":
        raise ValueError("Use a new dynamics_v2 config with this runner")
    # Resolve one fixed time condition from the cache, then validate every sample.
    if cfg.get("dynamics_fs") is None or cfg.get("dynamics_fs_type") is None:
        files = sorted(Path(cfg["train_cache"]).glob("*.pt"))
        if not files:
            raise ValueError("No dynamics cache found; export WMA predictions first")
        first = torch.load(files[0], map_location="cpu", weights_only=True)
        for key in ("dynamics_fs", "dynamics_fs_type"):
            if key not in first:
                raise ValueError(f"Cache lacks {key}; regenerate dynamics cache")
            if cfg.get(key) is None:
                cfg[key] = first[key]
        print("Dynamics time condition:", cfg["dynamics_fs_type"], cfg["dynamics_fs"])
    if cfg["dynamics_fs_type"] not in ("fps", "frame_stride") or cfg["dynamics_fs"] <= 0:
        raise ValueError("Invalid dynamics time condition")
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
        print("Resume checkpoint:" if args.resume else "Initialization checkpoint:",
              args.resume or args.init_from, flush=True)
        saved = torch.load(args.resume or args.init_from, map_location="cpu", weights_only=True)
        if saved["config"].get("implementation") != "dynamics_v2":
            raise ValueError("Legacy checkpoints are not compatible; train this experiment from scratch")
        if saved["config"]["model"] != cfg["model"]:
            raise ValueError("Checkpoint model configuration mismatch")
        model.load_state_dict(saved["model"], strict=True)
        if args.resume:
            if saved["config"].get("finetune_scope", "full") != "full":
                raise ValueError("Cannot resume removed restricted training; use --init-from to start full-task training")
            if saved["config"].get("left_grasp_action_weight", 0) and "left_grasp_lift_seconds" in saved["config"]:
                raise ValueError("The old phase-label objective was removed; use --init-from for a new run")
            for key in ("adaptive_distillation", "adaptive_temperature", "teacher_weight", "future_weight",
                        "adaptive_data", "data_weight", "data_temperature",
                        "dynamics_teacher_weight", "dynamics_real_weight", "dynamics_fs", "dynamics_fs_type", "adaptive_dynamics", "dynamics_temperature", "initial_action_weight",
                        "first_action_weight", "gripper_action_weight", "gripper_transition_weight", "first_step_gripper_transitions",
                        "gripper_boundary_sampling_weight", "checkpoint_metric", "left_grasp_action_weight"):
                if saved["config"].get(key) != cfg.get(key):
                    raise ValueError(f"Resume setting changed: {key}; use --init-from for a new experiment")
            if saved["config"].get("stage","joint") != stage:
                raise ValueError("Use --init-from to change training stage")
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            start, best = saved["epoch"]+1, saved.get("best", float("inf"))
    train_data = FeatureCache(cfg["train_cache"],cfg,True)
    val_data = FeatureCache(cfg["val_cache"],cfg)
    sampler = balanced_sampler(train_data,cfg["dataset_weights"]) if "dataset_weights" in cfg else None
    boundary_weight = cfg.get("gripper_boundary_sampling_weight", 1.0)
    if (isinstance(boundary_weight, bool) or not isinstance(boundary_weight, (int, float))
            or not math.isfinite(boundary_weight) or boundary_weight < 1):
        raise ValueError("gripper_boundary_sampling_weight must be finite and >=1")
    if boundary_weight > 1:
        if sampler is not None:
            raise ValueError("Boundary sampling and dataset_weights cannot be combined")
        weights = torch.ones(len(train_data), dtype=torch.double)
        for index, path in enumerate(train_data.files):
            sample = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            grip = sample["action"][:, 14:16]
            valid = sample["action_mask"][:, 14:16].bool()
            first_change = (((grip[0] >= .5) != (sample["state"][14:16] >= .5)) & valid[0]).any()
            # Include the open frames just before a transition as hard negatives,
            # so extra closing supervision does not simply encourage early closure.
            near_change = (((grip[1:4] >= 0) != (grip[:3] >= 0)) & valid[1:4] & valid[:3]).any()
            if first_change or near_change:
                weights[index] = boundary_weight
        sampler = WeightedRandomSampler(weights, len(train_data), replacement=True,
                                        generator=torch.Generator().manual_seed(cfg.get("seed", 42)))
        print("Boundary sampling:", int((weights > 1).sum()), "/", len(weights),
              "samples; weight", boundary_weight, flush=True)
    train = DataLoader(train_data,batch_size=cfg["batch_size"],sampler=sampler,shuffle=sampler is None)
    val = DataLoader(val_data, batch_size=cfg["batch_size"])
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
            if cfg.get("checkpoint_metric") == "control_score":
                values["control_score"] = checkpoint_score(values, cfg)
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
                    row["macro"] = {
                        key: sum(group[key] for group in groups.values() if key in group)
                             / sum(key in group for group in groups.values())
                        for key in set().union(*(group.keys() for group in groups.values()))}
            print(json.dumps(row),flush=True)
            with (output/"metrics.jsonl").open("a",encoding="utf-8") as file:
                file.write(json.dumps(row)+"\n")
        score = checkpoint_score(values, cfg)
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
