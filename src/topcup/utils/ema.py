import copy
import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        """
        Exponential Moving Average tracker for PyTorch models.

        Args:
            model (torch.nn.Module): the model to track.
            decay (float): decay rate (typically 0.99 - 0.9999).
        """
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

        # Ensure EMA model is on the same device as the original model
        device = next(model.parameters()).device
        self.ema_model.to(device)

        # Disable gradients for EMA model
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update EMA model parameters using the current model.

        Args:
            model (torch.nn.Module): the source model to update from.
        """
        msd = model.state_dict()
        for name, ema_param in self.ema_model.state_dict().items():
            model_param = msd[name].detach()

            # Skip if shape mismatch or not a float tensor
            if not model_param.shape == ema_param.shape:
                continue
            if not torch.is_floating_point(ema_param):
                continue

            # Ensure model_param is on same device
            model_param = model_param.to(ema_param.device)
            ema_param.mul_(self.decay).add_(model_param, alpha=1.0 - self.decay)

    def to(self, device):
        """Move EMA model to the specified device."""
        self.ema_model.to(device)

    def state_dict(self):
        """Get EMA model's state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict into EMA model."""
        self.ema_model.load_state_dict(state_dict)

    def eval(self):
        """Set EMA model to eval mode."""
        self.ema_model.eval()

    def train(self):
        """Set EMA model to train mode."""
        self.ema_model.train()