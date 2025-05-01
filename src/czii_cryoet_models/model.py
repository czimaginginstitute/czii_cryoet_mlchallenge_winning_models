import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import copy
from collections import defaultdict
from czii_cryoet_models.modules.unet import FlexibleUNet
from czii_cryoet_models.modules.utils import (
    count_parameters, 
    human_format,
    to_ce_target,
)
from czii_cryoet_models.loss.dense_cross_entropy import DenseCrossEntropy
from czii_cryoet_models.data.augmentation import Mixup
from czii_cryoet_models.postprocess.utils import (
    TARGET_CLASSES,
    TARGET_SIGMAS,
    sliding_window, 
    mean_std_renormalization, 
    decode_detections_with_nms,
    postprocess_scores_offsets_into_submission,
    postprocess_pipeline
)

class LitNet(pl.LightningModule):
    def __init__(
        self,
        classes: list,
        class_weights: np.ndarray,
        backbone_args: dict,
        lvl_weights: np.ndarray,
        mixup_beta: float = 1.0,
        mixup_p: float = 1.0,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.classes = classes
        self.n_classes = len(classes)
        self.model = FlexibleUNet(**backbone_args)
        self.class_weights = torch.from_numpy(class_weights).float()
        self.lvl_weights = torch.from_numpy(lvl_weights).float()
        self.loss_fn = DenseCrossEntropy(class_weights=self.class_weights)
        self.mixup = Mixup(mixup_beta)
        self.mixup_p = mixup_p
        self.learning_rate = learning_rate
        self.epoch_metrics = {"train_loss": [], "val_loss": [], "val_score": []}

        print(f'Net parameters: {human_format(count_parameters(self))}')


    def forward(self, x, y=None):
        if self.training and y is not None and torch.rand(1).item() < self.mixup_p:
            x, y = self.mixup(x, y)

        out = self.model(x)  # list of outputs at different scales
        outputs = {}

        if y is not None:
            # Compute multiscale loss
            ys = [F.adaptive_max_pool3d(y, item.shape[-3:]) for item in out]
            losses = torch.stack([
                self.loss_fn(out[i], to_ce_target(ys[i]))[0] for i in range(len(out))
            ])

            lvl_weights = self.lvl_weights.to(losses.device)
            loss = (losses * lvl_weights).sum() / lvl_weights.sum()

            outputs['loss'] = loss

        if not self.training:
            outputs['logits'] = out[-1]  # final output at full resolution

        # del batch
        # torch.cuda.empty_cache()

        return outputs

    
    def training_step(self, batch):
        outputs = self(batch["input"], batch["target"])
        self.log('train_loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True)
        return outputs['loss']
    
    
    def validation_step(self, batch):
        print(batch['input'].shape) # MetaTensor (B, 1, D, H, W)
        #targets = batch['target']
        metas = batch["meta"]

        # Sliding window inference, get logits per tomogram
        pred, val_loss = sliding_window(
            inputs=batch["input"],
            predictor=self,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            n_classes=7,
            overlap=(0.0, 0.0, 0.0),  # no overlap for validation inference
            z_scale=[0.5, 0.5, 0.5],
            verbose=False,
        )
        D, H, W = batch["input"].shape[-3:]
        f_beta = postprocess_pipeline(pred, metas, D, H, W)
        self.log('val_score', f_beta, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics[f"train_loss"].item()
        self.epoch_metrics[f"train_loss"].append(avg_loss)


    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics[f"val_loss"].item()
        avg_score = self.trainer.callback_metrics[f"val_score"].item()
        self.epoch_metrics[f"val_loss"].append(avg_loss)
        self.epoch_metrics[f"val_score"].append(avg_score)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
