import glob
import sys
import math
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import List
import monai
from torch.utils.data import DataLoader
from czii_cryoet_models.model import LitNet
from copick.impl.filesystem import CopickRootFSSpec
from czii_cryoet_models.data.copick_dataset import CopickDataset, TrainDataset
from czii_cryoet_models.data.augmentation import train_aug, get_basic_transform_list


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


def get_args():
    parser = argparse.ArgumentParser(
        description = "3D image segmentation",
    )
    parser.add_argument("-c", "--copick_config", help="copick config file path")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size for data loader")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("-d", "--debug", action='store_true', help="debugging True/ False")
    parser.add_argument("-p", "--pretrained_weights", type=str, default="", help="Pretrained weights file path. Default is None.")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs. Default is 100.")
    parser.add_argument("--pixelsize", type=float, default=10.012, help="Pixelsize. Default is 10.012A.")
    parser.add_argument("--distributed", type=bool, default=False, help="Distributed training, default is False.")
    parser.add_argument("-i", "--data_folder", type=str, default="./data", help="data folder for training")
    parser.add_argument("-t", "--train_df", type=str, default="train_folded_v1.csv", help="dataframe file containing label localizations")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="output dir for saving checkpoints")
    return parser.parse_args()


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
        self.train_dataset = TrainDataset(
            copick_root = self.copick_root,
            transforms = train_aug,
            run_names = ['TS_5_4', 'TS_6_4', 'TS_6_6', 'TS_69_2', 'TS_73_6', 'TS_86_3', 'TS_99_9'],
            classes = ['apo-ferritin','beta-amylase','beta-galactosidase','ribosome','thyroglobulin','virus-like-particle'],      
            pixelsize = 10.012,
            n_aug=40
        )

        self.val_dataset = CopickDataset(
            copick_root = self.copick_root,
            run_names = ['TS_5_4', 'TS_6_4'],
            classes = ['apo-ferritin','beta-amylase','beta-galactosidase','ribosome','thyroglobulin','virus-like-particle'],  
            transforms = monai.transforms.Compose(get_basic_transform_list(["input"])),    
            pixelsize = 10.012
        )
        # train_size = int(0.7 * len(dataset))
        # val_size = len(dataset) - train_size
        # print(f'full_dataset {len(dataset)}, train set {train_size}, val set {val_size}')

        # # Set random seed for deterministic dataset split
        # torch.manual_seed(42)
        # self.train_dataset, self.val_dataset = random_split(
        #     dataset, [train_size, val_size]
        # )
        
        
        
        
        # print(f'train_dataset1\n{len(self.train_dataset)}')
        # self.train_ds = TrainDataset(self.train_dataset)
        # print(f'train_dataset2\n{len(self.train_ds)}')

    def train_dataloader(self):
        print(f'self.train_dataset {len(self.train_dataset)}')
        train_dataloader = DataLoader(
            self.train_dataset,   # 1112*4
            #sampler=sampler,
            #shuffle=(sampler is None),
            batch_size=8, #self.batch_size,  # 8
            num_workers=8,  # one worker per batch
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last= True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return train_dataloader

    def val_dataloader(self):
        print(f'self.val_dataset {len(self.val_dataset)}')
        val_dataloader = DataLoader(
            self.val_dataset,   # 1112*4
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
        return val_dataloader



if __name__ == "__main__":
    args = get_args()
    copick_root = CopickRootFSSpec.from_file(args.copick_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_module = DataModule(copick_root=copick_root, batch_size=args.batch_size)

    backbone_args = dict(
        spatial_dims=3,    
        in_channels=1,
        out_channels=6,
        backbone='resnet34',
        pretrained=False
    )

    model = LitNet(        
        classes = ['apo-ferritin','beta-amylase','beta-galactosidase','ribosome','thyroglobulin','virus-like-particle'],
        class_weights = np.array([256,256,256,256,256,256,1]),
        backbone_args = backbone_args,
        lvl_weights = np.array([0, 0, 0, 1]),
    )

    # Define Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        mode="min",
        save_top_k=1,
        dirpath=f"{args.output_dir}/checkpoints/",
        filename="best_model"
    )

    # Logger
    logger = CSVLogger(save_dir=f"{args.output_dir}/logs/", name="training_logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=2,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    # Train Model
    trainer.fit(model, data_module)



