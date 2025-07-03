import torch
from torch.utils.data import DataLoader
from czii_cryoet_models.model import LitNet
from copick.impl.filesystem import CopickRootFSSpec
from czii_cryoet_models.data.copick_dataset import CopickDataset
from czii_cryoet_models.data.augmentation import get_basic_transform_list
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import monai
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description = "3D image segmentation inference",
    )
    parser.add_argument("-c", "--copick_config", help="copick config file path")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size for data loader")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("-d", "--debug", action='store_true', help="debugging True/ False")
    parser.add_argument("-p", "--pretrained_weights", type=str, default="", help="Pretrained weights file path. Default is None.")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs. Default is 100.")
    parser.add_argument("--pixelsize", type=float, default=10.012, help="Pixelsize. Default is 10.012A.")
    parser.add_argument("--distributed", type=bool, default=False, help="Distributed training, default is False.")
    parser.add_argument("-t", "--train_df", type=str, default="train_folded_v1.csv", help="dataframe file containing label localizations")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="output dir for saving checkpoints")
    return parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def collate_fn(batch):
    keys = batch[0].keys()
    batch_dict = {key:torch.cat([b[key] for b in batch]) for key in keys}
    return batch_dict


def inference_collate_fn(batch):
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
            batch_size: int = 1,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.copick_root = copick_root

    def setup(self, stage=None):    # stage='train' only gets called for training
        self.predict_dataset = CopickDataset(
            copick_root = self.copick_root,
            run_names = ['TS_5_4', 'TS_6_4'],
            transforms = monai.transforms.Compose(get_basic_transform_list(["input"])),    
            pixelsize = 10.012
        )

    def predict_dataloader(self):
        print(f'self.inference_dataset {len(self.predict_dataset)}')
        predict_dataloader = DataLoader(
            self.predict_dataset,   # 1112*4
            #sampler=sampler,
            #shuffle=(sampler is None),
            batch_size=2, #self.batch_size,  # 8
            num_workers=2,  # one worker per batch
            pin_memory=False,
            collate_fn=inference_collate_fn,
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
    data_module = DataModule(copick_root=copick_root, batch_size=args.batch_size)

    # Load model from checkpoint
    model = LitNet.load_from_checkpoint("/hpc/projects/group.czii/kevin.zhao/ml_challenge/winning_models/czii_cryoet_mlchallenge_models/output/checkpoints/best_model.ckpt")
    print(model)

    # Initialize trainer
    trainer = pl.Trainer(devices=1, accelerator="gpu")

    # Run prediction
    predictions = trainer.predict(model, datamodule=data_module)



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