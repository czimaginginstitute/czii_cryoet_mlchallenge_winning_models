import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import copy
import pandas as pd
from collections import defaultdict
from czii_cryoet_models.modules.unet import FlexibleUNet
from czii_cryoet_models.modules.utils import (
    count_parameters, 
    human_format,
    to_ce_target,
)
from czii_cryoet_models.loss.dense_cross_entropy import DenseCrossEntropy
from czii_cryoet_models.data.augmentation import Mixup
from czii_cryoet_models.postprocess.simple_pp import postprocess_pipeline_val
from czii_cryoet_models.postprocess.metric import calc_metric
from czii_cryoet_models.postprocess.utils import (
    sliding_window, 
    mean_std_renormalization, 
    decode_detections_with_nms,
    postprocess_scores_offsets_into_submission,
    postprocess_pipeline
)
from czii_cryoet_models.utils.utils import (
    get_optimizer,
    get_scheduler
)


class LitNet(pl.LightningModule):
    def __init__(
        self,
        nclasses: int,
        class_weights: np.ndarray,
        backbone_args: dict,
        lvl_weights: np.ndarray,
        mixup_beta: float = 1.0,
        mixup_p: float = 1.0,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = nclasses + 1  # +1 for background
        self.model = FlexibleUNet(**backbone_args)

        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.lvl_weights = torch.tensor(lvl_weights, dtype=torch.float32)

        self.loss_fn = DenseCrossEntropy(class_weights=self.class_weights)
        self.mixup = Mixup(mixup_beta)
        self.mixup_p = mixup_p
        self.learning_rate = learning_rate

        self.gt_dfs = []
        self.submission_dfs = []

    def forward(self, x, y=None):
        if self.training and y is not None and torch.rand(1).item() < self.mixup_p:
            x, y = self.mixup(x, y)

        out = self.model(x)

        if y is None:
            return {"logits": out[-1]}

        # Downsample ground truth to match each output level
        ys = [F.adaptive_max_pool3d(y, o.shape[-3:]) for o in out]
        losses = torch.stack([self.loss_fn(o, to_ce_target(ys[i]))[0] for i, o in enumerate(out)])
        lvl_weights = self.lvl_weights.to(losses.device)

        loss = (losses * lvl_weights).sum() / lvl_weights.sum()
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        out = self(batch["input"], batch["target"])
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        pred, val_loss = sliding_window(
            inputs=batch["input"],
            predictor=self,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            n_classes=self.n_classes,
            overlap=(0.0, 0.0, 0.0),
            z_scale=[0.5, 0.5, 0.5],
            verbose=False,
        )
        gt_df, submission_df = postprocess_pipeline_val(pred, batch["meta"])
        self.gt_dfs.append(gt_df)
        self.submission_dfs.append(submission_df)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        return val_loss

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     pred, _ = sliding_window(
    #         inputs=batch["input"],
    #         predictor=self,
    #         roi_size=(96, 96, 96),
    #         sw_batch_size=1,
    #         n_classes=self.n_classes,
    #         overlap=(0.21, 0.21, 0.21),
    #         z_scale=[0.5, 0.5, 0.5],
    #         verbose=False,
    #     )
    #     postprocess_pipeline_val(pred, batch["meta"])

    def on_validation_epoch_end(self):
        """Aggregate and log validation metrics."""
        if self.gt_dfs and self.submission_dfs:
            gt_df = pd.concat(self.gt_dfs, ignore_index=True)
            submission_df = pd.concat(self.submission_dfs, ignore_index=True)
            score = calc_metric(
                [
                    "apo-ferritin", "beta-amylase", "beta-galactosidase",
                    "ribosome", "thyroglobulin", "virus-like-particle"
                ],
                submission_df, gt_df
            )
            self.log("val_score", score["score"], on_epoch=True, prog_bar=True)

        self.gt_dfs.clear()
        self.submission_dfs.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

# class LitNet(pl.LightningModule):
#     def __init__(
#         self,
#         nclasses: int,
#         class_weights: np.ndarray,
#         backbone_args: dict,
#         lvl_weights: np.ndarray,
#         mixup_beta: float = 1.0,
#         mixup_p: float = 1.0,
#         learning_rate: float = 1e-3,
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         self.n_classes = nclasses+1 # add background class
#         self.model = FlexibleUNet(**backbone_args)
#         self.class_weights = torch.from_numpy(class_weights).float()
#         self.lvl_weights = torch.from_numpy(lvl_weights).float()
#         self.loss_fn = DenseCrossEntropy(class_weights=self.class_weights)
#         self.mixup = Mixup(mixup_beta)
#         self.mixup_p = mixup_p
#         self.learning_rate = learning_rate
#         self.epoch_metrics = {"avg_train_loss": [], "avg_val_loss": [], "avg_val_score": []}
#         self.gt_dfs = []
#         self.submission_dfs = []

#     def forward(self, x, y=None):
#         if self.training and y is not None and torch.rand(1).item() < self.mixup_p:  # random mix-up
#             x, y = self.mixup(x, y)

#         out = self.model(x)  # list of segmentation maps at different feature levels
#         outputs = {}

#         if y is not None:
#             # Compute multiscale loss
#             ys = [F.adaptive_max_pool3d(y, item.shape[-3:]) for item in out]  # downsample ground-truth point seg mask to match each output level size
#             losses = torch.stack([
#                 self.loss_fn(out[i], to_ce_target(ys[i]))[0] for i in range(len(out))
#             ])

#             lvl_weights = self.lvl_weights.to(losses.device)
#             loss = (losses * lvl_weights).sum() / lvl_weights.sum()

#             outputs['loss'] = loss

#         if not self.training:
#             outputs['logits'] = out[-1]  # output the segmentation map at the last level (1, 7, 48, 48, 48)

#         return outputs

    
#     def training_step(self, batch):
#         outputs = self(batch["input"], batch["target"])
#         self.log('train_loss', outputs['loss'], on_step=True, on_epoch=True, prog_bar=True)
#         return outputs['loss']
    
    
#     def validation_step(self, batch):
#         metas = batch["meta"]
#         # Sliding window inference, get logits per tomogram
#         pred, val_loss = sliding_window(
#             inputs=batch["input"],
#             predictor=self,
#             roi_size=(96, 96, 96),
#             sw_batch_size=1,
#             n_classes=self.n_classes,
#             overlap=(0.0, 0.0, 0.0),  # no overlap for validation inference
#             z_scale=[0.5, 0.5, 0.5],
#             verbose=False,
#         )
#         # pred before postprocess: torch.Size([7, 315, 315, 92])
#         # W H D torch.Size([2, 1, 630, 630, 184])
#         gt_df, submission_df = postprocess_pipeline_val(pred, metas)
#         self.gt_dfs.append(gt_df)
#         self.submission_dfs.append(submission_df)
#         #self.log('val_score', f_beta, prog_bar=True, on_step=False, on_epoch=True)
#         self.log('val_loss', val_loss, prog_bar=True, on_step=True, on_epoch=True)
#         return val_loss
    

#     def predict_step(self, batch):
#         print(batch['input'].shape) # MetaTensor (B, 1, D, H, W)
#         #targets = batch['target']
#         metas = batch["meta"]

#         # Sliding window inference, get logits per tomogram
#         pred, predic_loss = sliding_window(
#             inputs=batch["input"],
#             predictor=self,
#             roi_size=(96, 96, 96),
#             sw_batch_size=1,
#             n_classes=self.n_classes,
#             overlap=(0.21, 0.21, 0.21),  # no overlap for validation inference
#             z_scale=[0.5, 0.5, 0.5],
#             verbose=False,
#         )
#         D, H, W = batch["input"].shape[-3:]
#         postprocess_pipeline_val(pred, metas)
#         #postprocess_pipeline(pred, metas, D, H, W, mode='inference')

    
#     def on_train_epoch_end(self):
#         avg_loss = self.trainer.callback_metrics[f"train_loss"].item()
#         self.epoch_metrics[f"avg_train_loss"].append(avg_loss)


#     def on_validation_epoch_end(self):
#         """
#         Args:
#             outputs: list of dicts from validation_step
#         """
#         final_gt_df = pd.concat(self.gt_dfs, ignore_index=True)
#         final_submission_df = pd.concat(self.submission_dfs, ignore_index=True)
#         score_dict = calc_metric(["apo-ferritin", "beta-amylase", "beta-galactosidase", "ribosome", "thyroglobulin", "virus-like-particle"], final_submission_df, final_gt_df)
#         self.log('val_score', score_dict["score"], prog_bar=True, on_step=False, on_epoch=True)
#         avg_loss = self.trainer.callback_metrics[f"val_loss"].item()
#         self.epoch_metrics[f"avg_val_loss"].append(avg_loss)
#         self.gt_dfs = []
#         self.submission_dfs = []

    
#     def configure_optimizers(self):
#         optimizer = get_optimizer(
#             self, 
#             self.learning_rate, 
#             sgd_momentum=0.0,
#             sgd_nesterov=False
#         )
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#         #scheduler = get_scheduler()

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": scheduler,
#             "monitor": "val_loss"
#         }
