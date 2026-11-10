"""Frozen Base features with optional action-conditioned WMA prediction."""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F


def encoder_weights(saved, prefix=""):
    weights = saved.get("state_dict", saved.get("module", saved))
    if prefix:
        if not all(k.startswith(prefix) for k in weights):
            raise ValueError("Checkpoint keys do not share state_dict_prefix")
        weights = {k[len(prefix):]:v for k,v in weights.items()}
    result = {}
    for name in ("embedder", "cond_stage_model"):
        tag = name + "."
        selected = {k[len(tag):]:v for k,v in weights.items() if k.startswith(tag)}
        if not selected:
            raise ValueError(f"Base checkpoint lacks {name} weights; refusing random encoder fallback")
        result[name] = selected
    return result


class BaseEncoders:
    def __init__(self, cfg):
        self.dynamics_enabled = bool(cfg.get("enable_dynamics", False))
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda:0"))
        self.image_batch_size = int(cfg.get("image_batch_size", 1))
        if self.image_batch_size < 1:
            raise ValueError("image_batch_size must be positive")
        if self.dynamics_enabled:
            from .teacher import Teacher
            self._teacher = Teacher(cfg)
            self.embedder = self._teacher.model.embedder
            self.cond_stage_model = self._teacher.model.cond_stage_model
            self.text_cache = {}
            self.provenance = {"mode": "base_encoders_with_dynamics",
                               "checkpoint": str(Path(cfg["checkpoint"]).resolve()),
                               "strict_full_weights": True, "teacher_actions": False,
                               "dynamics": "action_projector + sim_mode=True",
                               "ddim_steps": cfg.get("ddim_steps", 16)}
            return
        checkpoint = Path(cfg["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Set the actual Base checkpoint path: {checkpoint}")
        repo = Path(cfg["repo"]).resolve()
        sys.path.insert(0, str(repo / "src"))
        from omegaconf import OmegaConf
        from unifolm_wma.utils.utils import instantiate_from_config
        config = OmegaConf.load(cfg["model_config"])
        # Deserialize to CPU only. mmap reduces RAM pressure for modern torch archives.
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as exc:
            if "mmap" not in str(exc):
                raise
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = encoder_weights(saved, cfg.get("state_dict_prefix", ""))
        self.device = torch.device(cfg.get("device", "cuda:0"))
        self.image_batch_size = int(cfg.get("image_batch_size", 1))
        if self.image_batch_size < 1:
            raise ValueError("image_batch_size must be positive")
        expected = {"img_cond_stage_config":"FrozenOpenCLIPImageEmbedderV2",
                    "cond_stage_config":"FrozenOpenCLIPEmbedder"}
        for key, name in expected.items():
            spec = OmegaConf.to_container(config.model.params[key], resolve=True)
            if spec["target"].rsplit(".",1)[-1] != name:
                raise ValueError(f"Unsupported encoder target: {spec['target']}")
            params = spec.setdefault("params", {})
            # Instantiate architecture without fetching unrelated pretrained weights.
            params.update(version=None, device=str(self.device), freeze=True)
            encoder = instantiate_from_config(spec)
            role = "embedder" if key == "img_cond_stage_config" else "cond_stage_model"
            encoder.load_state_dict(weights.pop(role), strict=True)
            encoder.eval().requires_grad_(False).to(self.device)
            setattr(self, role, encoder)
        del saved, weights
        self.text_cache = {}
        self.provenance = {"mode":"base_encoders", "checkpoint":str(checkpoint),
                           "model_config":str(Path(cfg["model_config"]).resolve()),
                           "strict_encoder_weights":True, "teacher_actions":False}

    @torch.inference_mode()
    def features(self, images):
        chunks = []
        for batch in images.split(self.image_batch_size):
            tokens = self.embedder(batch.to(self.device)).float()[:,1:]
            side = int(tokens.shape[1] ** .5)
            if side * side != tokens.shape[1]:
                raise ValueError("Expected square patch grid")
            grid = tokens.transpose(1,2).reshape(len(batch),-1,side,side)
            chunks.append(F.adaptive_avg_pool2d(grid,(8,8)).flatten(2).transpose(1,2).cpu())
        return torch.cat(chunks)

    @torch.inference_mode()
    def text(self, instruction):
        if instruction not in self.text_cache:
            self.text_cache[instruction] = self.cond_stage_model.encode([instruction]).float().mean(1)[0].cpu()
        return self.text_cache[instruction].clone()

    @torch.inference_mode()
    def predict_future(self, obs_images, obs_states, actions, instruction, fs):
        """Normalized images [3,O,H,W], states [O,D], actions [16,A].

        Returns terminal predicted visual features in the same space as features().
        No real future observation is accepted by this interface.
        """
        if not self.dynamics_enabled:
            raise RuntimeError("Set teacher.enable_dynamics=true to load the frozen WMA")
        m = self._teacher.model
        horizon = m.model.diffusion_model.temporal_length
        if horizon != 16 or actions.shape != (horizon, m.agent_action_dim):
            raise ValueError("WMA sampler requires 16 actions with checkpoint action width")
        if obs_images.ndim != 4 or obs_images.shape[0] != 3:
            raise ValueError("obs_images must have shape [3, observations, height, width]")
        if obs_states.shape != (obs_images.shape[1], m.agent_state_dim):
            raise ValueError("Observation state/image history or state width mismatch")
        if obs_states.shape[0] != m.agent_state_pos_emb.shape[-2]:
            raise ValueError("Observation history must match checkpoint positional embeddings")
        for value in (obs_images, obs_states, actions):
            if not torch.isfinite(value).all():
                raise ValueError("Dynamics inputs must be finite normalized tensors")
        if int(fs) <= 0:
            raise ValueError("fps/frame_stride must be positive")
        images = obs_images.unsqueeze(0).to(self.device)
        states = obs_states.unsqueeze(0).to(self.device)
        action = actions.unsqueeze(0).to(self.device)
        flat = images.permute(0, 2, 1, 3, 4).reshape(-1, 3, images.shape[-2], images.shape[-1])
        encoded = m.encode_first_stage(flat)
        z = encoded.reshape(1, images.shape[2], *encoded.shape[1:]).permute(0, 2, 1, 3, 4)
        cond = {"c_concat": [z[:, :, -1:].repeat(1, 1, horizon, 1, 1)],
                "c_crossattn": [torch.cat((
                    m.state_projector(states) + m.agent_state_pos_emb,
                    m.action_projector(action) + m.agent_action_pos_emb,
                    m.get_learned_conditioning([instruction]),
                    m.image_proj_model(m.embedder(images[:, :, -1]))), dim=1)],
                "c_crossattn_action": [images[:, :, -m.n_obs_steps_acting:],
                                       states[:, -m.n_obs_steps_acting:], True, False]}
        latents, _, _, _ = self._teacher.sampler.sample(
            S=self.cfg.get("ddim_steps", 16), conditioning=cond, batch_size=1,
            shape=(z.shape[1], horizon, z.shape[3], z.shape[4]), eta=0., verbose=False,
            unconditional_guidance_scale=1.,
            fs=torch.tensor([fs], device=self.device, dtype=torch.long),
            timestep_spacing="uniform_trailing")
        video = m.decode_first_stage(latents)
        if video.shape[2] != horizon or not torch.isfinite(video).all():
            raise ValueError("Invalid WMA predicted video")
        return self.features(video[0, :, -1].unsqueeze(0))[0]
