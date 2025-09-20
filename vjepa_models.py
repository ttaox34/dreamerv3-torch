
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import networks
import tools
import time
import os

# Assuming vjepa2 submodule is in the project root
try:
    from vjepa2.hubconf import vjepa2_vit_giant
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


class VJEPAWorldModel(nn.Module):
    """
    World model based on V-JEPA 2 AC, replacing the RSSM.
    """
    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self._config = config
        self._step = step
        self._act_space = act_space
        self._use_amp = True if config.precision == 16 else False
        
        # Determine action space size
        if hasattr(act_space, 'n'):
            self._action_size = act_space.n
        else:
            self._action_size = act_space.shape[0]

        # --- V-JEPA Models ---
        self.vjepa_encoder = self._load_vjepa_encoder(config.vjepa_encoder_path, config.freeze_vjepa_encoder)
        self.ac_predictor = self._load_ac_predictor(config.vjepa_ac_path, config.freeze_vjepa_predictor)

        # --- Action Adapter ---
        self.action_adapter = nn.Sequential(
            nn.Linear(self._action_size, 512),
            nn.ReLU(),
            nn.Linear(512, self._original_action_dim)
        )

        # --- DreamerV3 Heads ---
        # The feature size is the output dimension of the V-JEPA encoder
        feat_size = 1408 # For ViT-g
        self.heads = nn.ModuleDict()
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
            "model", self.heads.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._adapter_opt = tools.Optimizer(
            "adapter", self.action_adapter.parameters(), lr=config.vjepa['adapter_lr'], eps=config.vjepa['adapter_eps'], clip=config.vjepa['adapter_grad_clip'],
            wd=config.vjepa['adapter_wd'], opt=config.opt, use_amp=self._use_amp)

        print("VJEPA World Model Initialized.")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def _load_vjepa_encoder(self, path, freeze):
        print(f"Loading V-JEPA encoder from {path}")
        # Hardcoding ViT-g for now, can be made configurable
        encoder = vjepa2_vit_giant(pretrained=False)[0] # Create model instance
        
        if self._config.vjepa.get('use_dummy_models', False):
            print("Using dummy V-JEPA encoder.")
            return encoder

        if path and Path(path).exists():
            try:
                ckpt = torch.load(path, map_location='cpu')
                # The vitg.pt checkpoint is a dictionary, not a raw state dict
                if 'encoder' in ckpt:
                    encoder_ckpt = ckpt['encoder']
                elif 'model' in ckpt:
                    encoder_ckpt = ckpt['model']
                else:
                    encoder_ckpt = ckpt
                # Clean keys from prefixes
                cleaned_ckpt = {}
                for k, v in encoder_ckpt.items():
                    k = k.replace('module.', '')
                    k = k.replace('backbone.', '')
                    cleaned_ckpt[k] = v
                encoder.load_state_dict(cleaned_ckpt, strict=True)
                print("Encoder weights loaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not load encoder weights: {e}")

        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
            print("V-JEPA encoder is frozen.")

        # Move to device
        encoder.to(self._config.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for V-JEPA encoder via DataParallel.")
            encoder = nn.DataParallel(encoder)

        return encoder

    def _load_ac_predictor(self, path, freeze):
        print(f"Loading V-JEPA AC predictor from {path}")

        # The checkpoint expects specific dimensions. We hardcode them here.
        original_action_dim = 7
        original_state_dim = 7
        original_extrinsics_dim = 6
        self._original_action_dim = original_action_dim

        # Instantiate the model with the correct dimensions BEFORE loading weights
        predictor = vit_ac_predictor(
            img_size=self._config.size,
            patch_size=16,
            num_frames=self._config.batch_length,
            embed_dim=1408, 
            predictor_embed_dim=1024, 
            depth=24, 
            num_heads=16,
            use_rope=True, 
            action_embed_dim=original_action_dim,
            state_embed_dim=original_state_dim,
            extrinsics_embed_dim=original_extrinsics_dim,
            use_activation_checkpointing=self._config.vjepa['checkpointing'])

        if self._config.vjepa.get('use_dummy_models', False):
            print("Using dummy AC predictor.")
        elif path and Path(path).exists():
            try:
                full_ckpt = torch.load(path, map_location='cpu')
                pred_ckpt = full_ckpt['predictor']
                pred_ckpt = {k.replace('module.', ''): v for k, v in pred_ckpt.items()}
                predictor.load_state_dict(pred_ckpt, strict=True)
                print("AC Predictor weights loaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not load predictor weights: {e}")
        else:
            print("Warning: Predictor checkpoint not found. Using randomly initialized predictor.")

        if freeze:
            for param in predictor.parameters():
                param.requires_grad = False
            predictor.eval()
            print("V-JEPA AC predictor is fully frozen.")

        # Move to device
        predictor.to(self._config.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for V-JEPA AC predictor via DataParallel.")
            predictor = nn.DataParallel(predictor)
        
        return predictor

    def _train(self, data):
        data = self.preprocess(data)

        with tools.CPUTimeRecording("wm_train_encode"):
            with torch.no_grad():
                with torch.cuda.amp.autocast(self._use_amp):
                    z_patches = self.encode(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                B, T, H, W, C = data['image'].shape
                tubelet_size = 2
                patch_size = 16
                T_patches = T // tubelet_size
                H_patches = H // patch_size
                W_patches = W // patch_size
                D = z_patches.shape[-1]

                num_spatial_patches = H_patches * W_patches
                z_seq_patches = z_patches.view(B, T_patches, num_spatial_patches, D)

                actions_sub = data['action'][:, ::tubelet_size]
                B, T_sub, _ = actions_sub.shape
                dummy_states = torch.zeros(B, T_sub, 7, device=actions_sub.device)

                context_len = self._config.vjepa['pred_context_len']
                predictor_input = z_seq_patches[:, -context_len-1:-1].flatten(1, 2)

                actions_for_pred = actions_sub[:, -context_len-1:-1]
                states_for_pred = dummy_states[:, -context_len-1:-1]

                adapted_actions = self.action_adapter(actions_for_pred)

                with tools.CPUTimeRecording("wm_train_ac_predictor"):
                    z_pred_patches = self.ac_predictor(predictor_input, adapted_actions, states_for_pred)

                z_target_patches = z_seq_patches[:, -context_len:].flatten(1, 2).detach()
                pred_loss = F.l1_loss(z_pred_patches, z_target_patches)

                z_seq_agg = z_seq_patches.mean(dim=2)
                feats = z_seq_agg.detach()
                
                reward_data = data['reward'][:, ::tubelet_size]
                cont_data = data['cont'][:, ::tubelet_size]

                head_losses = {}
                pred_reward = self.heads['reward'](feats)
                head_losses['reward'] = -pred_reward.log_prob(reward_data).mean()

                pred_cont = self.heads['cont'](feats)
                head_losses['cont'] = -pred_cont.log_prob(cont_data).mean()

                adapter_loss = pred_loss
                model_loss = head_losses['reward'] + head_losses['cont']

            with tools.CPUTimeRecording("wm_train_adapter_opt"):
                metrics = self._adapter_opt(adapter_loss, self.action_adapter.parameters())
            with tools.CPUTimeRecording("wm_train_model_opt"):
                metrics.update(self._model_opt(model_loss, self.heads.parameters()))

        metrics.update({f'{name}_loss': loss.item() for name, loss in head_losses.items()})
        metrics['vjepa_pred_loss'] = pred_loss.item()

        start_z_agg = z_seq_patches[:, 0].mean(dim=1).detach()
        start_state = {'feat': start_z_agg, 'patches': z_seq_patches[:, 0].detach()}
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
        
        # Adapt the action to the predictor's expected dimension
        with torch.no_grad(): # Adapter is trained in the _train step
            adapted_action = self.action_adapter(action)

        # Reshape action to (B, 1, A_size) for the ac_predictor
        action_reshaped = adapted_action.unsqueeze(1) # (B, 1, A_size)
        dummy_state = torch.zeros(B, 1, 7, device=action_reshaped.device)

        # Predict next patches
        next_z_patches = self.ac_predictor(z_patches, action_reshaped, dummy_state)
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
