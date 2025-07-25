import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import json
import copy
import pandas as pd
from pathlib import Path
from czii_cryoet_models.modules.unet import FlexibleUNet
from czii_cryoet_models.modules.utils import (
    to_ce_target,
)
from czii_cryoet_models.loss.dense_cross_entropy import DenseCrossEntropy
from czii_cryoet_models.data.augmentation import Mixup
from czii_cryoet_models.utils.ema import ModelEMA
from czii_cryoet_models.postprocess.simple_pp import postprocess_pipeline_val, postprocess_pipeline_inference
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
        backbone_args: dict,
        lvl_weights: np.ndarray,
        class_loss_weights: dict={},
        mixup_beta: float = 1.0,
        mixup_p: float = 1.0,
        learning_rate: float = 1e-3,
        output_dir: str = './output/jobs/job_0/',
        ema_decay: float = 0.999,
        ema_start_epoch: int = 5,
        particle_ids: dict = {},
        particle_radius: dict = {},
        particle_weights: dict = {},
        score_thresholds: dict = {},
    ):
        super().__init__()

        # Model setup
        self.n_classes = nclasses + 1  # +1 for background
        self.model = FlexibleUNet(**backbone_args)
        self.ema = ModelEMA(self.model, decay=ema_decay)
        self.ema_start_epoch = ema_start_epoch
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)

        # Class weights setup
        weights = [0.0] * nclasses  # suppress missing classes
        for particle_name, weight in class_loss_weights.items():
            if particle_name in particle_ids:
                weights[particle_ids[particle_name]] = weight
        weights.append(1.0)  # background class
        self.class_loss_weights = torch.tensor(weights, dtype=torch.float32)

        # Loss and training utilities
        self.lvl_weights = torch.tensor(lvl_weights, dtype=torch.float32)
        self.loss_fn = DenseCrossEntropy(class_loss_weights=self.class_loss_weights)
        self.mixup = Mixup(mixup_beta)
        self.mixup_p = mixup_p

        # Particle metadata
        self.particle_ids = copy.deepcopy(particle_ids)
        self.particle_radius = copy.deepcopy(particle_radius)
        self.particle_weights = copy.deepcopy(particle_weights)
        self.score_thresholds = copy.deepcopy(score_thresholds)

        # Metrics tracking
        self.avg_train_losses = []
        self.train_losses = []
        self.avg_val_losses = []
        self.val_losses = []
        self.gt_dfs = []
        self.submission_dfs = []
        self.val_scores = []
        self.best_val_scores = {'score': 0.0}
        self.best_score_thresholds = {}

        # Save all arguments as hyperparameters
        self.save_hyperparameters()

    
    @property
    def description(self) -> str:
        desc_dict = {}
        for (k1, v1), (k2, v2), (k3, v3), (k4, v4) in zip(
            self.particle_ids.items(),
            self.particle_radius.items(),
            self.score_thresholds.items(),
            self.particle_weights.items()
        ):
            desc_dict.setdefault(k1, {})['channel_id'] = v1
            desc_dict.setdefault(k2, {})['radius'] = v2
            desc_dict.setdefault(k3, {})['score_threshold'] = v3
            desc_dict.setdefault(k4, {})['score_weight'] = v4
        
        desc_str = f"SegNet model predicting {self.n_classes-1} classes\n" 
        return f"{desc_str}\nPrediction details:\n{json.dumps(desc_dict, indent=2)}"
    
    
    @classmethod
    def load_flexible_checkpoints(cls, checkpoint_path, pattern='*.ckpt', **kwargs):
        checkpoint_paths = [Path(p.strip()) for p in checkpoint_path.split(',')]
        loaded_models = []

        for path in checkpoint_paths:
            if path.is_file():
                loaded_models.append(cls.load_from_checkpoint(str(path), **kwargs))
            elif path.is_dir():
                ckpt_files = sorted(path.glob(pattern))
                if not ckpt_files:
                    raise FileNotFoundError(f"No checkpoints matching pattern '{pattern}' in: {path}")
                print(f"[INFO] Loading {len(ckpt_files)} checkpoints from {path}")
                for p in ckpt_files:
                    loaded_models.append(cls.load_from_checkpoint(str(p), **kwargs))
            else:
                raise FileNotFoundError(f"Invalid path: {path}")

        if not loaded_models:
            raise FileNotFoundError("No valid checkpoints found.")

        return loaded_models
    
    
    @classmethod
    def ensemble_from_checkpoints(cls, checkpoint_path, pattern='*.ckpt', **kwargs):
        models = cls.load_flexible_checkpoints(checkpoint_path, pattern=pattern, **kwargs)
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

    
    def training_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        if batch is None:
            print(f"Skipping training batch {batch_idx} as it was None (empty after collation).")
            self.log('skipped_train_batches', 1, on_step=False, on_epoch=True, reduce_fx=torch.sum)
            return None # Return None to skip this batch in Lightning's training loop
        
        out = self(batch["input"], batch["target"])
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(out["loss"].item())
        if self.current_epoch >= self.ema_start_epoch:
            self.ema.update(self.model)
        
        return out["loss"]


    def validation_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        if batch is None:
            print(f"Skipping validation batch {batch_idx} as it was None (empty after collation).")
            self.log('skipped_val_batches', 1, on_step=False, on_epoch=True, reduce_fx=torch.sum) # Optional: log skipped batches
            return None # Return None to skip this batch for validation metrics and logging

        pred, val_loss = sliding_window(inputs=batch["input"], predictor=self, n_classes=self.n_classes)
        gt_df, submission_df = postprocess_pipeline_val(pred, batch["meta"])
        self.gt_dfs.append(gt_df)
        self.submission_dfs.append(submission_df)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_losses.append(val_loss)
        
        return val_loss


    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        if batch is None:
            print(f"Skipping prediction batch {batch_idx} as it was None (empty after collation).")
            return None # Return None to skip this batch in the prediction output
       
        with torch.inference_mode():
            print(f"[INFO] GPU {self.device} | Using ensemble of {len(self.models)} models." if hasattr(self, 'models') else "[INFO] Single model inference")

            # Input batch for this GPU
            img = batch["input"]                        # Shape: [B, C, D, H, W]
            img_flipped = torch.flip(img, [2, 3, 4])    # TTA
            preds = []

            # Get models to run (either ensemble or self)
            models = getattr(self, "models", [self])

            for model in models:
                model.eval()

                # Use model on same device as input (avoid .cuda())
                p1, _ = sliding_window(inputs=img, predictor=model, n_classes=self.n_classes)
                p2, _ = sliding_window(inputs=img_flipped, predictor=model, n_classes=self.n_classes)
                p2 = torch.flip(p2, [1, 2, 3])  # Flip back

                p_avg = (p1 + p2) / 2
                preds.append(p_avg)

            pred = torch.stack(preds).mean(dim=0)

            # Postprocess for this GPU's batch only
            if batch['dataset_type'] == 'copick' and batch['has_ground_truth']:
                gt_df, submission_df = postprocess_pipeline_val(pred, batch["meta"])
                self.gt_dfs.append(gt_df)
            else:
                submission_df = postprocess_pipeline_inference(pred, batch["meta"])
            self.submission_dfs.append(submission_df)
     
    
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
            scores, ths = calc_metric(
                submission_df, 
                gt_df,
                score_thresholds=self.score_thresholds,
                particle_radius = self.particle_radius,
                particle_weights = self.particle_weights
            )
            self.log("val_score", scores['score'], on_epoch=True, prog_bar=True)
            self.val_scores.append(scores)
            if scores['score'] >= self.best_val_scores['score']:
                self.best_val_scores = scores
                self.best_score_thresholds = ths

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_file = self.output_dir / 'metrics.json'
        data = dict()
        data['avg_train_loss'] = self.avg_train_losses
        data['avg_val_loss'] = self.avg_val_losses
        data['val_scores'] = self.val_scores
        data['best_val_score'] = self.best_val_scores
        data['best_score_thresholds'] = self.best_score_thresholds

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
                score, ths = calc_metric(
                    submission_df,
                    gt_df,
                    score_thresholds=self.score_thresholds,
                    particle_radius = self.particle_radius,
                    particle_weights = self.particle_weights,
                    output_dir=self.output_dir 
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

