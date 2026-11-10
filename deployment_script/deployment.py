"""In-process student inference shared by Isaac Lab and the optional HTTP adapter."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from lightweight_policy.base_encoders import BaseEncoders
from lightweight_policy.model import StudentPolicy
from lightweight_policy.model_dynamics import DynamicsStudentPolicy
from lightweight_policy.action_schema import load_schema

NAMES = ["k"+side+joint for side in ("Left","Right") for joint in
         ("ShoulderPitch","ShoulderRoll","ShoulderYaw","Elbow","WristRoll","WristPitch","WristYaw")] + ["kLeftGripper","kRightGripper"]

class Policy:
    def __init__(self, cfg, base_dir=None):
        base = Path(base_dir or Path.cwd()).resolve()
        def resolve(value):
            path = Path(value).expanduser()
            return str(path.resolve() if path.is_absolute() else (base / path).resolve())
        cfg = dict(cfg)
        for key in ("checkpoint", "export_config"):
            cfg[key] = resolve(cfg[key])
        self.cfg = cfg
        export = json.loads(Path(cfg["export_config"]).read_text(encoding="utf-8"))
        entries = export["datasets"]["train"]
        if isinstance(entries, dict): entries = [entries]
        entry = next(p for p in entries if p["dataset_name"] == cfg["dataset_name"])
        schema = load_schema(resolve(export["schema_sources"][cfg["dataset_name"]]))
        if any(schema[k]["names"] != NAMES for k in ("action","observation.state")):
            raise ValueError("This adapter requires exact G1 14-arm + left/right gripper order")
        if entry["normalization_mode"] != "min_max" or not entry["individual_normalization"]:
            raise ValueError("Adapter requires individual min_max normalization")
        if entry.get("spatial_transform") != "resize_center_crop":
            raise ValueError("Adapter requires resize_center_crop, matching export")
        saved = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
        spec = saved["config"]["model"]
        if spec.get("action_dim",16)!=16 or spec.get("state_dim",16)!=16 or spec.get("obs_steps",2)!=2:
            raise ValueError("Expected 16D actions/states and two observations")
        implementation = saved["config"].get("implementation")
        if implementation not in ("action_only_v1","dynamics_v2"):
            raise ValueError("Only action_only_v1 and dynamics_v2 checkpoints are supported")
        self.device = torch.device(cfg.get("device","cuda:0"))
        cls = DynamicsStudentPolicy if implementation=="dynamics_v2" else StudentPolicy
        self.model = cls(**spec).eval().requires_grad_(False).to(self.device)
        self.model.load_state_dict(saved["model"], strict=True)
        encoder_cfg = dict(export["teacher"], enable_dynamics=False, device=str(self.device))
        for key in ("repo", "checkpoint", "model_config"):
            encoder_cfg[key] = resolve(encoder_cfg[key])
        self.encoder = BaseEncoders(encoder_cfg)
        from unifolm_wma.data.utils import load_stats
        from unifolm_wma.data.normolize import Normalize, Unnormalize
        stats = load_stats(cfg["dataset_name"], None, resolve(entry["transition_dir"]))
        self.normalize = Normalize({"observation.state":[16]}, {"observation.state":"min_max"}, stats)
        self.unnormalize = Unnormalize({"action":[16]}, {"action":"min_max"}, stats)
        resolution=entry["resolution"]
        self.transform=transforms.Compose([transforms.Resize(min(resolution)), transforms.CenterCrop(resolution)])
        instruction=cfg["instruction"]
        if not instruction or "REPLACE" in instruction: raise ValueError("Set the instruction used for dataset export")
        self.text=self.encoder.text(instruction).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, images, state):
        """Accept two HWC uint8 RGB arrays and a raw 16D robot state."""
        if len(images) != 2:
            raise ValueError("Two RGB observations required")
        frames = []
        for image in images:
            image = np.asarray(image)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or not all(0 < n <= 4096 for n in image.shape):
                raise ValueError("Expected HWC uint8 RGB image")
            rgb = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
            frames.append(self.transform(rgb) / 127.5 - 1)
        state = torch.as_tensor(state, dtype=torch.float32, device="cpu")
        if state.shape != (16,) or not torch.isfinite(state).all():
            raise ValueError("Invalid state")
        normalized=self.normalize({"observation.state":state})["observation.state"]
        # Keep corrected physical finger readings within the model training range.
        if self.cfg.get("clip_gripper_state_to_training_range", True):
            normalized = normalized.clone()
            normalized[14:] = normalized[14:].clamp(-1.0, 1.0)
        visual=self.encoder.features(torch.stack(frames)).unsqueeze(0).to(self.device)
        action=self.model.predict(visual,self.text,normalized.unsqueeze(0).to(self.device))["action"][0].float().cpu()
        action=self.unnormalize({"action":action})["action"]
        if not torch.isfinite(action).all(): raise ValueError("Nonfinite policy output")
        return action.numpy()
