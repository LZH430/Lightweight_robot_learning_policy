"""Original UnifoLM teacher; run in its dependency environment."""
import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F


class Teacher:
    def __init__(self, cfg):
        path = Path(cfg["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(f"Replace the placeholder checkpoint with a real local file: {path}")
        sys.path.insert(0, str(Path(cfg["repo"]) / "src"))
        from omegaconf import OmegaConf
        from unifolm_wma.utils.utils import instantiate_from_config
        from unifolm_wma.models.samplers.ddim import DDIMSampler
        config = OmegaConf.load(cfg["model_config"])
        original_cwd = Path.cwd()
        try:
            os.chdir(Path(cfg["repo"]).resolve())
            self.model = instantiate_from_config(config.model)
        finally:
            os.chdir(original_cwd)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        weights = saved.get("state_dict", saved.get("module", saved))
        prefix = cfg.get("state_dict_prefix", "")
        if prefix:
            if not all(k.startswith(prefix) for k in weights):
                raise ValueError("Not every checkpoint key has the configured prefix")
            weights = {k[len(prefix):]: v for k, v in weights.items()}
        self.model.load_state_dict(weights, strict=True)
        del weights, saved
        self.model.eval().requires_grad_(False).to(cfg.get("device", "cuda:0"))
        self.sampler = DDIMSampler(self.model)
        self.cfg = cfg
        if self.model.model.conditioning_key != "hybrid":
            raise ValueError("This exporter supports hybrid decision-making teachers only")
        if getattr(self.model, "interp_mode", False):
            raise ValueError("Interpolation teachers need a separate observation-only adapter")

    @torch.no_grad()
    def features(self, images):
        # [frames, 3, H, W], [-1, 1]. Drop CLS, then spatially pool patches.
        tokens = self.model.embedder(images.cuda()).float()[:, 1:]
        side = int(tokens.shape[1] ** .5)
        if side * side != tokens.shape[1]:
            raise ValueError("Expected square patch grid from FrozenOpenCLIPImageEmbedderV2")
        grid = tokens.transpose(1, 2).reshape(len(images), -1, side, side)
        return F.adaptive_avg_pool2d(grid, (8, 8)).flatten(2).transpose(1, 2).cpu()

    @torch.no_grad()
    def text(self, instruction):
        return self.model.get_learned_conditioning([instruction]).float().mean(1)[0].cpu()

    @torch.no_grad()
    def actions(self, obs_images, obs_states, instruction, fs):
        m = self.model
        images = obs_images.unsqueeze(0).cuda()  #
        states = obs_states.unsqueeze(0).cuda()
        horizon = m.model.diffusion_model.temporal_length
        z = m.encode_first_stage(images)
        cond_image = m.image_proj_model(m.embedder(images[:, :, -1]))
        cond_state = m.state_projector(states) + m.agent_state_pos_emb
        text = m.get_learned_conditioning([instruction])
        cond = {"c_concat": [z[:, :, -1:].repeat(1, 1, horizon, 1, 1)],
                "c_crossattn": [torch.cat((cond_state, text, cond_image), dim=1)],
                "c_crossattn_action": [images[:, :, -m.n_obs_steps_acting:],
                                       states[:, -m.n_obs_steps_acting:]]}
        _, actions, _, _ = self.sampler.sample(
            S=self.cfg.get("ddim_steps", 16), conditioning=cond, batch_size=1,
            shape=(z.shape[1], horizon, z.shape[3], z.shape[4]),
            eta=0., verbose=False, unconditional_guidance_scale=1.,
            fs=torch.tensor([fs], device="cuda", dtype=torch.long),
            timestep_spacing="uniform_trailing")
        return actions[0].float().cpu()
