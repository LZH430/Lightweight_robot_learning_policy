"""Export WMAData clips and optional observation-only teacher actions."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from .action_schema import verify, normalization_sources, dataset_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mode = cfg.get("feature_backend", "teacher")
    if mode not in ("teacher", "base_encoders"):
        raise ValueError("feature_backend must be teacher or base_encoders")
    if mode == "base_encoders" and cfg.get("export_teacher_actions", True):
        raise ValueError("base_encoders requires export_teacher_actions=false")
    schemas = verify(cfg)
    stats_sources = normalization_sources(cfg)
    # Check placeholder before importing the original repository dependencies.
    if not Path(cfg["teacher"]["checkpoint"]).is_file():
        raise FileNotFoundError("Set teacher.checkpoint to your trained local checkpoint; checkpoint_url is only a placeholder")
    output = Path(cfg["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; export will not mix cache versions")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if mode == "base_encoders":
        from .base_encoders import BaseEncoders
        teacher = BaseEncoders(cfg["teacher"])
    else:
        from .teacher import Teacher  # Legacy full-teacher backend only.
        teacher = Teacher(cfg["teacher"])
    from unifolm_wma.data.wma_data import WMAData
    datasets = [(split, WMAData(**params)) for split, params in dataset_entries(cfg)]
    if {split for split, _ in datasets} != {"train", "val"}:
        raise ValueError("Provide train and val dataset definitions with separate trajectory CSV files")
    def episodes(dataset):
        # Group multiple camera rows by transition, not camera video.
        return {str(Path(dataset._get_transition_path(row)).resolve())
                for _, row in dataset.metadata.iterrows()}
    episode_sets = {split: set().union(*(episodes(d) for s, d in datasets if s == split))
                    for split in ("train", "val")}
    if episode_sets["train"] & episode_sets["val"]:
        raise ValueError("Train/val contain overlapping transition episodes")
    metadata = {"format_version": 3, "action_schemas": schemas, "normalization_sources": stats_sources,
                "action_mapping": "preserve source order; exact named dimensions required",
                "units_verified": False, "checkpoint_semantics_verified": False, "config": cfg, "samples": [],
                "features": "FrozenOpenCLIPImageEmbedderV2 patches without CLS, pooled 8x8; text mean over tokens",
                "teacher_action_weights": "ordinary action_unet, not dp_ema_model" if cfg.get("export_teacher_actions", True) else None,
                "encoder_provenance": getattr(teacher, "provenance", {"mode":mode}),
                "normalization": "WMAData individual normalization + unified dimension mapping"}
    for split, dataset in datasets:
        if not dataset.individual_normalization:
            raise ValueError("Enable individual_normalization for export")
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)
        allowed_paths = {str(Path(dataset._get_video_path(row)).resolve())
                         for _, row in dataset.metadata.iterrows()}
        for index in range(len(dataset)):
            for clip in range(cfg.get("clips_per_split", {}).get(split, cfg.get("clips_per_episode", 1))):
                sample = dataset[index]
                # WMAData can fall back to another row on decode failure; ensure same split.
                if str(Path(sample["path"]).resolve()) not in allowed_paths:
                    raise ValueError("Dataset returned a video outside the configured split")
                obs = sample["observation.image"]
                states = sample["observation.state"]
                images = torch.cat((obs.permute(1, 0, 2, 3),
                                    sample["video"][:, -1].unsqueeze(0)), dim=0)
                features = teacher.features(images)
                record = {"visual": features[:-1], "future": features[-1],
                          "text": teacher.text(sample["instruction"]),
                          "state": states[-1].float(), "action": sample["action"].float(),
                          "action_mask": sample["action_mask"].float()}
                if cfg.get("export_teacher_actions", True):
                    fs_key = "fps" if teacher.model.fps_condition_type == "fps" else "frame_stride"
                    record["teacher_action"] = teacher.actions(obs, states, sample["instruction"], sample[fs_key])
                    if record["teacher_action"].shape != record["action"].shape:
                        raise ValueError("Teacher and demonstration action shapes differ")
                if getattr(teacher, "dynamics_enabled", False):
                    fs_key = "fps" if teacher._teacher.model.fps_condition_type == "fps" else "frame_stride"
                    record["teacher_future"] = teacher.predict_future(
                        obs, states, record["action"], sample["instruction"], sample[fs_key])
                    record["dynamics_action"] = record["action"].clone()
                    record["dynamics_fs"] = int(sample[fs_key])
                    record["dynamics_fs_type"] = fs_key
                    record["dynamics_target_frame"] = int(record["action"].shape[0] - 1)
                    if record["teacher_future"].shape != record["future"].shape:
                        raise ValueError("Teacher and real future feature shapes differ")
                identity = hashlib.sha256(obs.numpy().tobytes() + record["action"].numpy().tobytes()).hexdigest()
                expected = len(schemas[dataset.dataset_name]["action"]["names"])
                if record["action"].shape[-1] != expected or not (record["action_mask"] == 1).all():
                    raise ValueError("Named full-dimensional actions require matching width and all-valid mask")
                record["dataset_name"] = dataset.dataset_name
                record["normalization_sha256"] = stats_sources[split][dataset.dataset_name]["sha256"]
                record["schema_dataset"] = dataset.dataset_name
                record["schema_sha256"] = schemas[dataset.dataset_name]["sha256"]
                record["episode_id"] = str(sample["path"])
                record["sample_id"] = identity
                filename = f"{dataset.dataset_name}_{index:06d}_{clip:03d}.pt"
                torch.save(record, folder / filename)
                metadata["samples"].append({"split": split, "dataset_name": dataset.dataset_name, "file": filename, "video": sample["path"],
                                             "sample_id": identity, "frame_stride": int(sample["frame_stride"])})
                print(split, filename, tuple(record["visual"].shape), flush=True)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Export complete. Match student visual_dim/text_dim/tokens to the saved tensors.")


if __name__ == "__main__":
    main()
