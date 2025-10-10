import torch
from torch import nn

class DenseCrossEntropy(nn.Module):
    def __init__(self, class_loss_weights=None):
        super(DenseCrossEntropy, self).__init__()

        self.class_loss_weights = class_loss_weights
    
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float)

        # Compute per-class negative log-likelihood
        loss = -logprobs * target
        
        # Average the loss over batch, depth, height, and width 
        class_losses = loss.mean((0,2,3,4))
        if self.class_loss_weights is not None:
            loss = (class_losses * self.class_loss_weights.to(class_losses.device)).sum() #/ class_loss_weights.sum() 
        else:
            
            loss = class_losses.sum()
        return loss, class_losses