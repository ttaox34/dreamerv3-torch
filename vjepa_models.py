
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import math

import networks
import tools

# Assuming vjepa2 submodule is in the project root
try:
    from vjepa2.hubconf import vjepa2_vit_large
    from vjepa2.src.models.ac_predictor import vit_ac_predictor
except ImportError:
    print("Warning: Could not import from vjepa2 submodule. Make sure it's initialized.")
    # Define dummy classes if import fails to allow for initial setup
    class DummyModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.layer = nn.Linear(1,1)
        def forward(self, *args, **kwargs):
            return torch.zeros(1)
    vjepa2_vit_giant = lambda **kwargs: DummyModel()
    vit_ac_predictor = lambda **kwargs: DummyModel()


class _IdentityPredictor(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class VJEPAWorldModel(nn.Module):
    """
    World model based on V-JEPA 2 AC, replacing the RSSM.
    """
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self._config = config
        self._step = step
        self._act_space = act_space
        self._use_amp = config.precision == 16
        self._vjepa_cfg = getattr(config, "vjepa", {})

        # Determine action space size
        if hasattr(act_space, "n"):
            self._action_size = act_space.n
        else:
            self._action_size = act_space.shape[0]

        # Model hyper-parameters (override-able from config)
        self._tubelet_size = int(self._vjepa_cfg.get("tubelet_size", 2))
        self._patch_size = int(self._vjepa_cfg.get("patch_size", 16))
        self._context_len = int(self._vjepa_cfg.get("pred_context_len", 16))
        self._encoder_arch = self._vjepa_cfg.get("encoder_arch", "vit_large")
        self._encoder_embed_dim = int(self._vjepa_cfg.get("encoder_embed_dim", 1024))
        raw_pred_dim = self._vjepa_cfg.get("predictor_embed_dim")
        self._predictor_embed_dim_override = raw_pred_dim not in (None, 0)
        self._predictor_embed_dim = (
            int(raw_pred_dim)
            if self._predictor_embed_dim_override
            else self._encoder_embed_dim
        )
        self._predictor_depth = int(self._vjepa_cfg.get("predictor_depth", 24))
        self._predictor_heads = int(self._vjepa_cfg.get("predictor_heads", 16))
        self._predictor_is_frame_causal = self._vjepa_cfg.get(
            "predictor_is_frame_causal", True
        )
        self._predictor_use_rope = self._vjepa_cfg.get("use_rope", True)
        self._predictor_use_silu = self._vjepa_cfg.get("use_predictor_silu", False)
        self._predictor_wide_silu = self._vjepa_cfg.get("predictor_wide_silu", True)
        self._predictor_drop_rate = self._vjepa_cfg.get("predictor_drop_rate", 0.0)
        self._predictor_attn_drop_rate = self._vjepa_cfg.get(
            "predictor_attn_drop_rate", 0.0
        )
        self._predictor_drop_path_rate = self._vjepa_cfg.get(
            "predictor_drop_path_rate", 0.0
        )
        self._predictor_use_checkpointing = self._vjepa_cfg.get("checkpointing", False)
        requested_checkpointing = self._vjepa_cfg.get("gradient_checkpointing")
        if requested_checkpointing is not None:
            if requested_checkpointing and config.freeze_vjepa_predictor:
                print(
                    "Gradient checkpointing requested but predictor is frozen; disabling checkpointing to maintain gradients."
                )
                self._predictor_use_checkpointing = False
            else:
                self._predictor_use_checkpointing = bool(requested_checkpointing)
        elif config.freeze_vjepa_predictor and self._predictor_use_checkpointing:
            print("Disabling V-JEPA predictor activation checkpointing to preserve gradients.")
            self._predictor_use_checkpointing = False
        self._predictor_use_extrinsics = self._vjepa_cfg.get("use_extrinsics", False)

        # Action conditioning dims (default to the environment space)
        cfg_action_dim: Optional[int] = self._vjepa_cfg.get("action_embed_dim")
        self._action_embed_dim = (
            int(cfg_action_dim) if cfg_action_dim not in (None, 0) else self._action_size
        )
        raw_state_dim = self._vjepa_cfg.get("state_embed_dim")
        self._state_embed_dim = (
            int(raw_state_dim)
            if raw_state_dim not in (None, -1)
            else self._action_embed_dim
        )
        self._action_dim_warning_emitted = False
        self._state_dim_warning_emitted = False

        # --- V-JEPA Models ---
        self.vjepa_encoder = self._load_vjepa_encoder(
            config.vjepa_encoder_path, config.freeze_vjepa_encoder
        )
        self.ac_predictor = self._load_ac_predictor(
            config.vjepa_ac_path, config.freeze_vjepa_predictor
        )

        # --- DreamerV3 Heads ---
        self.heads = nn.ModuleDict()
        feat_size = self._encoder_embed_dim
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            layers=config.reward_head["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            layers=config.cont_head["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )

        # --- Optimizers ---
        self._model_opt = tools.Optimizer(
            "model",
            self.heads.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

        self.reward_manager = tools.RewardManager(config)

        print("VJEPA World Model Initialized.")
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def _load_vjepa_encoder(self, path, freeze):
        print(f"Loading V-JEPA encoder from {path}")

        if self._config.vjepa.get("use_dummy_models", False):
            print("Using dummy V-JEPA encoder.")
            encoder = nn.Identity()
            self._encoder_embed_dim = self._encoder_embed_dim or 1024
            return encoder

        if self._encoder_arch != "vit_large":
            raise ValueError(f"Unsupported V-JEPA encoder arch: {self._encoder_arch}")

        encoder = vjepa2_vit_large(pretrained=False)[0]

        if path and Path(path).exists():
            try:
                ckpt = torch.load(path, map_location="cpu")
                if "encoder" in ckpt:
                    encoder_ckpt = ckpt["encoder"]
                elif "model" in ckpt:
                    encoder_ckpt = ckpt["model"]
                else:
                    encoder_ckpt = ckpt
                cleaned_ckpt = {}
                for k, v in encoder_ckpt.items():
                    k = k.replace("module.", "")
                    k = k.replace("backbone.", "")
                    cleaned_ckpt[k] = v
                encoder.load_state_dict(cleaned_ckpt, strict=True)
                print("Encoder weights loaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not load encoder weights: {e}")
        else:
            print("Warning: Encoder checkpoint not found. Using randomly initialized encoder.")

        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
            print("V-JEPA encoder is frozen.")

        encoder.to(self._config.device)
        if torch.cuda.device_count() > 1:
            print(
                f"Using {torch.cuda.device_count()} GPUs for V-JEPA encoder via DataParallel."
            )
            encoder = nn.DataParallel(encoder)

        actual_embed_dim = self._extract_embed_dim(encoder)
        if actual_embed_dim != self._encoder_embed_dim:
            print(
                f"Updating encoder feature dim from {self._encoder_embed_dim} to {actual_embed_dim}."
            )
            self._encoder_embed_dim = actual_embed_dim
            if not self._predictor_embed_dim_override:
                self._predictor_embed_dim = actual_embed_dim

        return encoder

    def _extract_embed_dim(self, encoder_module) -> int:
        module = encoder_module.module if isinstance(encoder_module, nn.DataParallel) else encoder_module
        if hasattr(module, "embed_dim"):
            return int(module.embed_dim)
        if hasattr(module, "encoder") and hasattr(module.encoder, "embed_dim"):
            return int(module.encoder.embed_dim)
        if hasattr(module, "head") and hasattr(module.head, "weight"):
            return int(module.head.weight.shape[1])
        return int(self._encoder_embed_dim)

    def _match_feature_dim(self, tensor, target_dim, warn_attr, label):
        current_dim = tensor.shape[-1]
        if current_dim == target_dim:
            return tensor
        if current_dim > target_dim:
            if label == "action":
                remainder = tensor[..., target_dim:]
                # Common case: padded zeros beyond the expected button slots.
                if remainder.abs().max().item() < 1e-6:
                    return tensor[..., :target_dim]
                combos = current_dim
                bits = int(round(math.log2(combos)))
                if 2**bits == combos and bits == target_dim:
                    indices = torch.argmax(tensor, dim=-1)
                    bit_indices = torch.arange(bits, device=tensor.device)
                    multi_hot = ((indices.unsqueeze(-1) >> bit_indices) & 1).to(tensor.dtype)
                    return multi_hot
            if not getattr(self, warn_attr):
                print(
                    f"Truncating {label} features from {current_dim} to {target_dim} to match V-JEPA predictor."
                )
                setattr(self, warn_attr, True)
            return tensor[..., :target_dim]
        pad = target_dim - current_dim
        if not getattr(self, warn_attr):
            print(
                f"Padding {label} features from {current_dim} to {target_dim} with zeros to match V-JEPA predictor."
            )
            setattr(self, warn_attr, True)
        return F.pad(tensor, (0, pad))

    def _prepare_states(self, data, reference_tensor):
        state_key = self._vjepa_cfg.get("state_tensor_key")
        candidate = None
        for key in filter(None, [state_key, "state", "game_state"]):
            if key and key in data:
                candidate = data[key]
                break
        if candidate is None:
            B, T, _ = reference_tensor.shape
            return torch.zeros(
                B, T, self._state_embed_dim, device=reference_tensor.device
            )
        if candidate.dim() == 2:
            candidate = candidate.unsqueeze(1)
        candidate = candidate[:, :: self._tubelet_size]
        candidate = candidate.to(reference_tensor.device).to(reference_tensor.dtype)
        return self._match_feature_dim(
            candidate, self._state_embed_dim, "_state_dim_warning_emitted", "state"
        )

    def _load_ac_predictor(self, path, freeze):
        print(f"Loading V-JEPA AC predictor from {path}")

        if self._config.vjepa.get("use_dummy_models", False):
            print("Using dummy AC predictor.")
            predictor = _IdentityPredictor()
            return predictor

        predictor = vit_ac_predictor(
            img_size=self._config.size,
            patch_size=self._patch_size,
            num_frames=self._config.batch_length,
            tubelet_size=self._tubelet_size,
            embed_dim=self._encoder_embed_dim,
            predictor_embed_dim=self._predictor_embed_dim,
            depth=self._predictor_depth,
            num_heads=self._predictor_heads,
            use_rope=self._predictor_use_rope,
            use_activation_checkpointing=self._predictor_use_checkpointing,
            is_frame_causal=self._predictor_is_frame_causal,
            action_embed_dim=self._action_embed_dim,
            drop_rate=self._predictor_drop_rate,
            attn_drop_rate=self._predictor_attn_drop_rate,
            drop_path_rate=self._predictor_drop_path_rate,
            use_silu=self._predictor_use_silu,
            wide_silu=self._predictor_wide_silu,
            use_extrinsics=self._predictor_use_extrinsics,
        )

        if path and Path(path).exists():
            try:
                full_ckpt = torch.load(path, map_location="cpu")
                if "predictor" in full_ckpt:
                    pred_ckpt = full_ckpt["predictor"]
                elif "state_dict" in full_ckpt:
                    pred_ckpt = full_ckpt["state_dict"]
                else:
                    pred_ckpt = full_ckpt
                pred_ckpt = {k.replace("module.", ""): v for k, v in pred_ckpt.items()}
                predictor.load_state_dict(pred_ckpt, strict=True)
                print("AC predictor weights loaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not load predictor weights: {e}")
        else:
            print(
                "Warning: Predictor checkpoint not found. Using randomly initialized predictor."
            )

        if freeze:
            for param in predictor.parameters():
                param.requires_grad = False
            predictor.eval()
            print("V-JEPA AC predictor is fully frozen.")

        predictor.to(self._config.device)
        if torch.cuda.device_count() > 1:
            print(
                f"Using {torch.cuda.device_count()} GPUs for V-JEPA AC predictor via DataParallel."
            )
            predictor = nn.DataParallel(predictor)

        return predictor

    def _train(self, data):
        data = self.preprocess(data)

        with tools.CPUTimeRecording("wm_train_encode"):
            with torch.inference_mode():
                with torch.cuda.amp.autocast(self._use_amp):
                    z_patches = self.encode(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                B, T, H, W, _ = data["image"].shape
                if T % self._tubelet_size != 0:
                    raise ValueError(
                        f"Batch length {T} must be divisible by tubelet_size {self._tubelet_size}."
                    )
                if H % self._patch_size != 0 or W % self._patch_size != 0:
                    raise ValueError(
                        f"Image spatial dims {(H, W)} must be divisible by patch_size {self._patch_size}."
                    )

                T_patches = T // self._tubelet_size
                H_patches = H // self._patch_size
                W_patches = W // self._patch_size
                D = z_patches.shape[-1]
                num_spatial_patches = H_patches * W_patches
                z_seq_patches = z_patches.view(B, T_patches, num_spatial_patches, D)

                actions_sub = data["action"][:, :: self._tubelet_size]
                actions_for_pred = self._match_feature_dim(
                    actions_sub,
                    self._action_embed_dim,
                    "_action_dim_warning_emitted",
                    "action",
                )
                states_for_pred = self._prepare_states(data, actions_for_pred)

                max_context = min(self._context_len, z_seq_patches.shape[1] - 1)
                if max_context <= 0:
                    raise ValueError(
                        "Context length must be at least 1 after accounting for available frames."
                    )
                predictor_input = z_seq_patches[:, -max_context - 1 : -1].flatten(1, 2)
                actions_ctx = actions_for_pred[:, -max_context - 1 : -1]
                states_ctx = states_for_pred[:, -max_context - 1 : -1]

                with tools.CPUTimeRecording("wm_train_ac_predictor"):
                    z_pred_patches = self.ac_predictor(
                        predictor_input, actions_ctx, states_ctx
                    )
                z_target_patches = (
                    z_seq_patches[:, -max_context:].flatten(1, 2).detach()
                )
                pred_loss = F.l1_loss(z_pred_patches, z_target_patches)

                z_seq_agg = z_seq_patches.mean(dim=2)
                feats = z_seq_agg.detach()

                reward_data = self.reward_manager.compute_reward(
                    base_reward=data["reward"],
                    done=data["is_terminal"],
                    obs=data["image"],
                )
                reward_data = reward_data[:, :: self._tubelet_size]
                cont_data = data["cont"][:, :: self._tubelet_size]

                head_losses = {}
                pred_reward = self.heads["reward"](feats)
                head_losses["reward"] = -pred_reward.log_prob(reward_data).mean()

                pred_cont = self.heads["cont"](feats)
                head_losses["cont"] = -pred_cont.log_prob(cont_data).mean()

                model_loss = head_losses["reward"] + head_losses["cont"]

            with tools.CPUTimeRecording("wm_train_model_opt"):
                metrics = self._model_opt(model_loss, self.heads.parameters())

        metrics.update({f"{name}_loss": loss.item() for name, loss in head_losses.items()})
        metrics["vjepa_pred_loss"] = pred_loss.item()

        start_z_agg = z_seq_patches[:, 0].mean(dim=1).detach()
        start_state = {
            "feat": start_z_agg,
            "patches": z_seq_patches[:, 0].detach(),
        }
        return start_state, {"feat": start_z_agg}, metrics

    def encode(self, obs):
        img = obs['image']
        # Handle both 4D (B, H, W, C) and 5D (B, T, H, W, C) inputs
        if img.dim() == 4:
            img = img.unsqueeze(1)  # Add time dimension for single-step obs
        
        # The V-JEPA patch embed kernel has a temporal size of 2.
        # If we have only one frame, we need to duplicate it.
        if img.size(1) == 1:
            img = img.repeat(1, 2, 1, 1, 1)

        # vjepa_encoder expects (B, C, T, H, W)
        img = img.permute(0, 4, 1, 2, 3)
        z = self.vjepa_encoder(img)
        return z

    def obs_step(self, prev_latent, prev_action, embed, is_first):
        # This method is called by the policy to get the current latent state.
        # With V-JEPA, the state is not recurrent. The new state is simply the
        # embedding of the current observation, which is passed in as `embed`.
        
        # If it's the first step of an episode, reset the latent state to zeros.
        if torch.any(is_first):
            # The latent state is the feature vector from the encoder.
            # For the first step, we can use a zero vector of the same shape.
            latent = torch.zeros_like(embed)
        else:
            latent = embed
        return latent

    def imagine_step(self, z_patches, action):
        # z_patches is (B, N_s, D). Predictor needs a sequence.
        # We treat the spatial patches as the sequence.
        B, N_s, D = z_patches.shape
        
        if action.dim() == 1:
            action = action.unsqueeze(0)
        action = action.to(z_patches.device).float()
        action = action.unsqueeze(1)
        action = self._match_feature_dim(
            action, self._action_embed_dim, "_action_dim_warning_emitted", "action"
        )

        state_tokens = torch.zeros(
            B, 1, self._state_embed_dim, device=z_patches.device, dtype=z_patches.dtype
        )

        next_z_patches = self.ac_predictor(z_patches, action, state_tokens)
        return next_z_patches

    def preprocess(self, obs):
        obs = {k: torch.tensor(v, device=self._config.device) for k, v in obs.items()}
        for key, value in obs.items():
            if value.dtype == torch.uint8:
                obs[key] = value.float() / 255.0 - 0.5
        if 'discount' in obs:
            obs['discount'] *= self._config.discount
        obs['cont'] = (1.0 - obs['is_terminal'].float()).unsqueeze(-1)
        return obs

    def get_feat(self, z):
        # z can be (B, N, D) or (H, B, N, D). We aggregate the patch dimension N.
        # The patch dimension is always the second to last.
        if z.dim() < 3:
            return z
        return z.mean(dim=-2)

    def on_episode_end(self):
        """Called at the end of an episode to reset components like the reward generator."""
        self.reward_manager.reset_generator()

    @property
    def feature_dim(self):
        return self._encoder_embed_dim
