import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import copy, json
import pandas as pd
from pathlib import Path
from czii_cryoet_models.modules.unet import FlexibleUNet
from czii_cryoet_models.modules.utils import (
    to_ce_target,
)
from czii_cryoet_models.loss.dense_cross_entropy import DenseCrossEntropy
from czii_cryoet_models.data.augmentation import Mixup
from czii_cryoet_models.postprocess.simple_pp import postprocess_pipeline_val
from czii_cryoet_models.postprocess.metric import calc_metric
from czii_cryoet_models.postprocess.utils import (
    sliding_window,
    get_final_submission 
)
from czii_cryoet_models.utils.utils import (
    get_optimizer,
)


class SegNet(pl.LightningModule):
    def __init__(
        self,
        nclasses: int,
        class_weights: np.ndarray,
        backbone_args: dict,
        lvl_weights: np.ndarray,
        mixup_beta: float = 1.0,
        mixup_p: float = 1.0,
        learning_rate: float = 1e-3,
        output_dir: str = './output/jobs/job_0/',
    ):
        super().__init__()
        self.save_hyperparameters()  # The parameters will be saved into the checkpoints.

        self.n_classes = nclasses + 1  # add the background channel
        self.model = FlexibleUNet(**backbone_args)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.lvl_weights = torch.tensor(lvl_weights, dtype=torch.float32)
        self.loss_fn = DenseCrossEntropy(class_weights=self.class_weights)
        self.mixup = Mixup(mixup_beta)
        self.mixup_p = mixup_p
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        
        self.score_thresholds=dict()
        self.score_thresholds["apo-ferritin"]=-1 #0.16 
        self.score_thresholds["beta-amylase"]=-1 #0.25 
        self.score_thresholds["beta-galactosidase"]=-1 #0.13
        self.score_thresholds["ribosome"]=-1 #0.19 
        self.score_thresholds["thyroglobulin"]=-1 #0.18
        self.score_thresholds["virus-like-particle"]=-1 #0.5
        
        self.gt_dfs = []
        self.submission_dfs = []
        self.inference_dfs = []

        self.avg_train_losses = []
        self.train_losses = []
        self.avg_val_losses = []
        self.val_losses = []
        self.avg_val_metrics = []

    
    @classmethod
    def load_flexible_checkpoints(cls, checkpoint_path, **kwargs):
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            return [cls.load_from_checkpoint(str(checkpoint_path), **kwargs)]

        elif checkpoint_path.is_dir():
            ckpt_files = sorted(checkpoint_path.glob("*.ckpt"))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoints found in directory: {checkpoint_path}")
            print(f"[INFO] Loading {len(ckpt_files)} checkpoints from {checkpoint_path}")
            return [cls.load_from_checkpoint(str(p), **kwargs) for p in ckpt_files]

        else:
            raise FileNotFoundError(f"Invalid path: {checkpoint_path}")
    
    
    @classmethod
    def ensemble_from_checkpoints(cls, checkpoint_path, **kwargs):
        models = cls.load_flexible_checkpoints(checkpoint_path, **kwargs)
        ensemble_model = models[0]           # Use the first model instance as the wrapper
        ensemble_model.models = models       # Inject all loaded models into `.models`
        return ensemble_model
    
    
    def forward(self, x, y=None):
        if self.training and y is not None and torch.rand(1).item() < self.mixup_p:
            x, y = self.mixup(x, y)

        out = self.model(x)

        if y is None:
            return {"logits": out[-1]} # use the penultimate logits

        # Downsample ground truth to match each output level
        ys = [F.adaptive_max_pool3d(y, o.shape[-3:]) for o in out]
        losses = torch.stack([self.loss_fn(o, to_ce_target(ys[i]))[0] for i, o in enumerate(out)])
        lvl_weights = self.lvl_weights.to(losses.device)

        loss = (losses * lvl_weights).sum() / lvl_weights.sum()
        return {"loss": loss}

    
    def training_step(self, batch, batch_idx):
        out = self(batch["input"], batch["target"])
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(out["loss"].item())
        return out["loss"]


    def validation_step(self, batch, batch_idx):
        pred, val_loss = sliding_window(inputs=batch["input"], predictor=self, n_classes=self.n_classes)
        gt_df, submission_df = postprocess_pipeline_val(pred, batch["meta"])
        self.gt_dfs.append(gt_df)
        self.submission_dfs.append(submission_df)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_losses.append(val_loss)
        return val_loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.inference_mode():
            print(f"[INFO] Using ensemble of {len(self.models)} models." if self.models else "[INFO] Single model inference")

            # TTA
            img = batch["input"]
            img_flipped = torch.flip(img, [2, 3, 4])
            preds = []

            models = self.models if self.models else [self]

            for model in models:
                model.eval().cuda()

                # Get predictions from original and flipped input
                p1, _ = sliding_window(inputs=img, predictor=model, n_classes=self.n_classes)
                p2, _ = sliding_window(inputs=img_flipped, predictor=model, n_classes=self.n_classes)

                # Flip p2 back
                p2 = torch.flip(p2, [1, 2, 3])

                # Average both predictions and store
                p_avg = (p1 + p2) / 2
                preds.append(p_avg)

            # Average across all models
            pred = torch.stack(preds).mean(dim=0)

            # Postprocess and accumulate
            gt_df, submission_df = postprocess_pipeline_val(pred, batch["meta"])
            self.gt_dfs.append(gt_df)
            self.submission_dfs.append(submission_df)

            # inference_df = postprocess_pipeline_val(pred, batch["meta"])
            # self.inference_dfs.append(inference_df)
    

    def on_train_epoch_end(self):
        self.avg_train_losses.append(sum(self.train_losses)/len(self.train_losses))
        self.train_losses = []
    

    def on_validation_epoch_end(self):
        """Aggregate and log validation metrics."""
        self.avg_val_losses.append(sum(self.val_losses)/len(self.val_losses))
        self.val_losses = []

        if self.gt_dfs and self.submission_dfs:
            gt_df = pd.concat(self.gt_dfs, ignore_index=True)
            submission_df = pd.concat(self.submission_dfs, ignore_index=True)
            score = calc_metric(
                submission_df, 
                gt_df,
                score_thresholds=self.score_thresholds,
            )
            self.log("val_score", score["score"], on_epoch=True, prog_bar=True)
            self.avg_val_metrics.append(score["score"])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_file = self.output_dir / 'metrics.json'
        data = dict()
        data['avg_train_loss'] = self.avg_train_losses
        data['avg_val_loss'] = self.avg_val_losses
        data['avg_val_metric'] = self.avg_val_metrics

        with json_file.open("w") as f:
            json.dump(data, f, indent=2)
        
        self.gt_dfs.clear()
        self.submission_dfs.clear()
    
    
    def on_predict_epoch_end(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.submission_dfs:
            submission_df = pd.concat(self.submission_dfs, ignore_index=True)
            # If ground truth is also available, compute metric
            if self.gt_dfs:
                gt_df = pd.concat(self.gt_dfs, ignore_index=True)
                score = calc_metric(
                    submission_df,
                    gt_df,
                    score_thresholds=self.score_thresholds
                )

            # Generate and save the final submission
            get_final_submission(
                submission_df,
                score_thresholds=self.score_thresholds,
                output_dir=str(self.output_dir)
            )

            self.gt_dfs.clear()
            self.submission_dfs.clear()


    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

