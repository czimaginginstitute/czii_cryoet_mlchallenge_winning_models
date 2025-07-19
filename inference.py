import torch
from torch.utils.data import DataLoader
from czii_cryoet_models.model import SegNet
from copick.impl.filesystem import CopickRootFSSpec
from czii_cryoet_models.data.copick_dataset import CopickDataset
from czii_cryoet_models.data.augmentation import get_basic_transform_list
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import monai
from pathlib import Path
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description = "3D image segmentation inference",
    )
    parser.add_argument("-c", "--copick_config", help="copick config file path")
    parser.add_argument("-ts", "--run_names", type=str, default="", help="Tomogram dataset run names")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, help="batch size for data loader")
    parser.add_argument("-p", "--pretrained_weights", type=str, default="", help="Pretrained weights file paths (use comma for multiple paths). Default is None.")
    parser.add_argument("-pa", "--pattern", type=str, default="*.ckpt", help="The key for pattern matching checkpoints. Default is *.ckpt")
    parser.add_argument("--pixelsize", type=float, default=10.012, help="Pixelsize. Default is 10.012A.")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="output dir for saving prediction results (csv).")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs for inference. Default is 1.")
    return parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def collate_fn(batch):
    keys = batch[0].keys()
    batch_dict = {key:torch.cat([b[key] for b in batch]) for key in keys}
    return batch_dict


def data_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values  # leave as list
    return collated


class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            copick_root = None,
            run_names = None,
            pixelsize = 10.012,
            batch_size: int = 1,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.copick_root = copick_root
        self.run_names = run_names
        self.pixelsize = pixelsize

    def setup(self, stage=None):    # stage='train' only gets called for training
        self.predict_dataset = CopickDataset(
            copick_root = self.copick_root,
            run_names = self.run_names,
            transforms = monai.transforms.Compose(get_basic_transform_list(["input"])),    
            pixelsize = self.pixelsize
        )

    def predict_dataloader(self):
        print(f'self.inference_dataset {len(self.predict_dataset)}')
        predict_dataloader = DataLoader(
            self.predict_dataset,   # 1112*4
            #sampler=sampler,
            #shuffle=(sampler is None),
            batch_size=self.batch_size, 
            num_workers=self.batch_size,  # one worker per batch
            pin_memory=False,
            collate_fn=data_collate_fn,
            drop_last= True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return predict_dataloader


if __name__ == "__main__":
    args = get_args()
    copick_root = CopickRootFSSpec.from_file(args.copick_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_module = DataModule(copick_root=copick_root, run_names=args.run_names.split(','), batch_size=args.batch_size, pixelsize=args.pixelsize)

    output_dir = Path(f'{args.output_dir}')
    print(f'making output dir {str(output_dir)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models from checkpoints
    ensemble_model = SegNet.ensemble_from_checkpoints(args.pretrained_weights, pattern=args.pattern)
    ensemble_model.output_dir = args.output_dir
    #ensemble_model.score_thresholds = {"apo-ferritin": 0.16, "beta-amylase": 0.25, "beta-galactosidase": 0.13, "ribosome": 0.19, "thyroglobulin": 0.18, "virus-like-particle": 0.5}
    ensemble_model.score_thresholds = {"apo-ferritin": 0.41, "beta-amylase": 0.36, "beta-galactosidase": 0.41, "ribosome": 0.19, "thyroglobulin": 0.295, "virus-like-particle": 0.24}

    # Initialize trainer
    trainer = pl.Trainer(devices=1, accelerator="gpu") if args.gpus == 1 else pl.Trainer(devices=args.gpus, accelerator="gpu", strategy="ddp")

    # Run prediction
    predictions = trainer.predict(ensemble_model, datamodule=data_module)




# # Paths to multiple checkpoints
# checkpoint_paths = [
#     "checkpoints/model1.ckpt",
#     "checkpoints/model2.ckpt",
#     "checkpoints/model3.ckpt"
# ]

# # Create inference dataset + dataloader
# inference_dataset = MyInferenceDataset(...)
# inference_loader = DataLoader(inference_dataset, batch_size=4, shuffle=False)

# # For storing all model predictions
# all_model_preds = []

# # Predict with each model
# for ckpt in checkpoint_paths:
#     print(f"Predicting with checkpoint: {ckpt}")
#     model = LitModel.load_from_checkpoint(ckpt)

#     trainer = Trainer(devices=1, accelerator="gpu", logger=False)
#     preds = trainer.predict(model, dataloaders=inference_loader)

#     # preds is a list of tensors → stack into one tensor (B, ...)
#     stacked = torch.cat(preds, dim=0)
#     all_model_preds.append(stacked)

# # Average predictions across models
# ensemble_preds = torch.stack(all_model_preds).mean(dim=0)  # shape: (B, ...)

# # Save if needed
# torch.save(ensemble_preds, "ensemble_predictions.pt")