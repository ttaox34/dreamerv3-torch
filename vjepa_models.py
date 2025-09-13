
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import networks
import tools

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

        # --- Optimizer ---
        self._model_opt = tools.Optimizer(
            "model", self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt, use_amp=self._use_amp)

        print("VJEPA World Model Initialized.")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def _load_vjepa_encoder(self, path, freeze):
        print(f"Loading V-JEPA encoder from {path}")
        # Hardcoding ViT-g for now, can be made configurable
        encoder = vjepa2_vit_giant(pretrained=False)[0] # Create model instance
        
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
        return encoder

    def _load_ac_predictor(self, path, freeze):
        print(f"Loading V-JEPA AC predictor from {path}")
        predictor = vit_ac_predictor(
            img_size=self._config.size,
            patch_size=16,
            num_frames=self._config.batch_length,
            embed_dim=1408, predictor_embed_dim=1024, depth=24, num_heads=16,
            use_rope=True, action_embed_dim=self._action_size,
            use_activation_checkpointing=False) # Disable checkpointing
        
        if path and Path(path).exists():
            try:
                full_ckpt = torch.load(path, map_location='cpu')
                pred_ckpt = full_ckpt['predictor']
                # Clean keys from 'module.' prefix if it exists
                pred_ckpt = {k.replace('module.', ''): v for k, v in pred_ckpt.items()}

                pred_ckpt.pop('action_encoder.weight', None)
                pred_ckpt.pop('action_encoder.bias', None)
                pred_ckpt.pop('state_encoder.weight', None)
                pred_ckpt.pop('state_encoder.bias', None)
                pred_ckpt.pop('extrinsics_encoder.weight', None)
                pred_ckpt.pop('extrinsics_encoder.bias', None)
                
                predictor.load_state_dict(pred_ckpt, strict=False)
                print("AC Predictor weights loaded successfully (excluding action/state encoders).")
            except Exception as e:
                print(f"ERROR: Could not load predictor weights: {e}")

        if freeze:
            for name, param in predictor.named_parameters():
                if 'action_encoder' not in name: # Keep action_encoder trainable
                    param.requires_grad = False
            predictor.train() # Keep it in train mode for dropout, etc. if needed, but grads are frozen
            print("V-JEPA AC predictor is frozen (except action_encoder).")
        
        return predictor

    def _train(self, data):
        data = self.preprocess(data)

        # Encode observations with the frozen encoder without tracking gradients.
        with torch.no_grad():
            with torch.cuda.amp.autocast(self._use_amp):
                z_patches = self.encode(data)

        # Now, with gradients enabled only for the trainable parts, compute losses.
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                B, T, H, W, C = data['image'].shape
                tubelet_size = 2
                patch_size = 16
                T_patches = T // tubelet_size
                H_patches = H // patch_size
                W_patches = W // patch_size
                D = z_patches.shape[-1]

                # Reshape to (B, T_patches, num_spatial_patches, D)
                num_spatial_patches = H_patches * W_patches
                z_seq_patches = z_patches.view(B, T_patches, num_spatial_patches, D)

                # Subsample actions to match temporal resolution of z
                actions_sub = data['action'][:, ::tubelet_size]
                dummy_states = torch.zeros_like(actions_sub)

                # Prepare inputs for the predictor.
                # The predictor expects a 3D tensor (B, SEQ, D) where SEQ is the flattened time and space dimensions.
                predictor_input = z_seq_patches[:, :-1].flatten(1, 2)

                # The actions and states should be a simple time sequence.
                actions_for_pred = actions_sub[:, :-1]
                states_for_pred = dummy_states[:, :-1]

                z_pred_patches = self.ac_predictor(predictor_input, actions_for_pred, states_for_pred)

                # V-JEPA prediction loss
                z_target_patches = z_seq_patches[:, 1:].flatten(1, 2).detach()
                pred_loss = F.l1_loss(z_pred_patches, z_target_patches)

                # DreamerV3 head losses (on aggregated features)
                z_seq_agg = z_seq_patches.mean(dim=2)
                feats = z_seq_agg.detach()
                
                reward_data = data['reward'][:, ::tubelet_size]
                cont_data = data['cont'][:, ::tubelet_size]

                head_losses = {}
                pred_reward = self.heads['reward'](feats)
                head_losses['reward'] = -pred_reward.log_prob(reward_data).mean()

                pred_cont = self.heads['cont'](feats)
                head_losses['cont'] = -pred_cont.log_prob(cont_data).mean()

                total_loss = pred_loss + head_losses['reward'] + head_losses['cont']

            metrics = self._model_opt(total_loss, self.parameters())

        metrics.update({f'{name}_loss': loss.item() for name, loss in head_losses.items()})
        metrics['vjepa_pred_loss'] = pred_loss.item()

        # Return start state for imagination
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
        
        # Reshape action to (B, 1, A_size) for the ac_predictor
        action_reshaped = action.unsqueeze(1) # (B, 1, A_size)
        dummy_state = torch.zeros_like(action_reshaped)

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
