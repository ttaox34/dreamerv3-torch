
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
            embed_dim=1408, predictor_embed_dim=1024, depth=24, num_heads=16,
            use_rope=True, action_embed_dim=self._action_size)
        
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
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # Encode observations
                z = self.encode(data)
                
                # Prepare actions for predictor (one-hot encoding)
                actions_onehot = F.one_hot(data['action'].long(), num_classes=self._action_size).float()
                
                # Predict next latent state
                dummy_states = torch.zeros_like(actions_onehot)
                z_pred = self.ac_predictor(z[:-1], actions_onehot[:-1], dummy_states[:-1])
                
                # V-JEPA prediction loss
                z_target = z[1:].detach()
                pred_loss = F.l1_loss(z_pred, z_target)

                # DreamerV3 head losses
                head_losses = {}
                feats = z.detach() # Use detached features for heads
                for name, head in self.heads.items():
                    pred = head(feats)
                    loss = -pred.log_prob(data[name])
                    head_losses[name] = loss.mean()

                total_loss = pred_loss + head_losses['reward'] + head_losses['cont']

            metrics = self._model_opt(total_loss, self.parameters())

        metrics.update({f'{name}_loss': loss.item() for name, loss in head_losses.items()})
        metrics['vjepa_pred_loss'] = pred_loss.item()
        
        # Return start state for imagination
        start_z = z.detach()
        return start_z, {"feat": start_z}, metrics

    def encode(self, obs):
        # obs['image'] is (B, T, H, W, C)
        # vjepa_encoder expects (B, C, T, H, W)
        img = obs['image'].permute(0, 4, 1, 2, 3)
        z = self.vjepa_encoder(img)
        return z

    def obs_step(self, prev_latent, prev_action, obs, is_first):
        # This method is called by the policy to get the current latent state.
        # With V-JEPA, the state is not recurrent in the same way as RSSM.
        # We simply encode the current observation.
        # prev_latent and prev_action are ignored, but kept for interface compatibility.
        
        # If it's the first step, create a zero latent tensor.
        if torch.any(is_first):
            batch_size = is_first.size(0)
            # This is a bit of a hack, we don't know the feature size without the encoder
            # Hardcoding 1408 for ViT-g
            latent = torch.zeros(batch_size, 1408, device=self._config.device)
        else:
            latent = self.encode(obs)
        return latent

    def imagine_step(self, z, action):
        # Used by ImagBehavior to unroll trajectories
        action_onehot = F.one_hot(action.long(), num_classes=self._action_size).float()
        dummy_state = torch.zeros_like(action_onehot)
        z_next = self.ac_predictor(z, action_onehot, dummy_state)
        return z_next

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
        # The feature is simply the latent state z
        return z
