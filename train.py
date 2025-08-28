import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import monai
from torch.utils.data import DataLoader
from czii_cryoet_models.model import SegNet
from czii_cryoet_models.data.utils import worker_init_fn, collate_fn, train_collate_fn
from copick.impl.filesystem import CopickRootFSSpec
from czii_cryoet_models.data.copick_dataset import CopickDataset, TrainDataset
from czii_cryoet_models.data.augmentation import train_aug, get_basic_transform_list
from czii_cryoet_models.postprocess.constants import ANGSTROMS_IN_PIXEL, CLASS_INDEX_TO_CLASS_NAME, TARGET_SIGMAS



def get_args():
    parser = argparse.ArgumentParser(
        description = "3D image segmentation",
    )
    parser.add_argument("-c", "--copick_config", help="copick config file path")
    parser.add_argument("-tts", "--train_run_names", type=str, default="", help="Tomogram dataset run names for training")
    parser.add_argument("-vts", "--val_run_names", type=str, default="", help="Tomogram dataset run names for validation")
    parser.add_argument("-rt", "--reconstruction_type", type=str, default="denoised", help="Tomogram reconstruction type. Default is denoised.")
    parser.add_argument("-u", "--user_id", type=str, default="curation", help="Needed for training, the user_id used for the ground truth picks.")
    parser.add_argument("-s", "--session_id", type=str, default="None", help="Needed for training, the session_id used for the ground truth picks. Default is None.")
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="batch size for data loader")
    parser.add_argument("-n", "--n_aug", type=int, default=1112, help="Data augmentation copy. Default is 1112.")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("-d", "--debug", action='store_true', help="debugging True/ False")
    parser.add_argument("-p", "--pretrained_weight", type=str, default="", help="One pretrained weights file path. Default is None.")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs. Default is 100.")
    parser.add_argument("--pixelsize", type=float, default=10.0, help="Pixelsize. Default is 10.0A.")
    parser.add_argument("--distributed", type=bool, default=False, help="Distributed training, default is False.")
    parser.add_argument("-i", "--data_folder", type=str, default="./data", help="data folder for training")
    parser.add_argument("-t", "--train_df", type=str, default="train_folded_v1.csv", help="dataframe file containing label localizations")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="output dir for saving checkpoints")
    parser.add_argument("-j", "--job_id", type=str, default="job_0", help="Unique id for each run")
    return parser.parse_args()


class DataModule(pl.LightningDataModule):
    def __init__(
            self, 
            copick_root = None,
            train_run_names = None,
            val_run_names = None,
            recon_type = 'denoised',
            user_id = 'curation',
            session_id = 'None',
            pixelsize = 10.012,
            batch_size: int = 1,
            n_aug: int = 1112,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.copick_root = copick_root
        self.train_run_names = train_run_names
        self.val_run_names = val_run_names
        self.recon_type = recon_type
        self.user_id = user_id
        self.session_id = session_id
        self.pixelsize = pixelsize
        self.n_aug = n_aug

    def setup(self, stage=None):    # stage='train' only gets called for training
        self.train_dataset = TrainDataset(
            copick_root = self.copick_root,
            transforms = train_aug,
            run_names = self.train_run_names, 
            pixelsize =  self.pixelsize,
            recon_type = self.recon_type,
            user_id = self.user_id,
            session_id = self.session_id,
            n_aug=self.n_aug,
            crop_radius = 2
        )

        self.val_dataset = CopickDataset(
            copick_root = self.copick_root,
            run_names = self.val_run_names,
            recon_type = self.recon_type,
            user_id = self.user_id,
            transforms = monai.transforms.Compose(get_basic_transform_list(["input"])),    
            pixelsize = self.pixelsize
        )

    def train_dataloader(self):
        print(f'train_dataset {len(self.train_dataset)}')
        train_dataloader = DataLoader(
            self.train_dataset,   # 1112*4
            #sampler=sampler,
            #shuffle=(sampler is None),
            batch_size=self.batch_size, #self.batch_size,  # 8
            num_workers=self.batch_size,  # one worker per batch
            pin_memory=False,
            collate_fn=train_collate_fn,
            drop_last= True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return train_dataloader

    def val_dataloader(self):
        print(f'val_dataset {len(self.val_dataset)}')
        val_dataloader = DataLoader(
            self.val_dataset,   # 1112*4
            #sampler=sampler,
            #shuffle=(sampler is None),
            batch_size=1, #self.batch_size
            num_workers=1, 
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last= True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return val_dataloader



if __name__ == "__main__":
    args = get_args()
    copick_root = CopickRootFSSpec.from_file(args.copick_config)
    copick_pickable_objects = []
    for obj in copick_root.pickable_objects:
        if obj.is_particle:
            if obj.radius:
                copick_pickable_objects.append(obj)
            else:
                print(f'Skipping {obj.name} in data loading becuase it does not have a radius in the copick configuration file.')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_module = DataModule(copick_root=copick_root, 
                             train_run_names=args.train_run_names.split(','), 
                             val_run_names=args.val_run_names.split(','), 
                             batch_size=args.batch_size, 
                             pixelsize=args.pixelsize,
                             recon_type=args.reconstruction_type,
                             user_id=args.user_id,
                             session_id=args.session_id,
                             n_aug=args.n_aug
                             )
    output_dir = Path(f'{args.output_dir}/jobs/{args.job_id}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'making output dir {str(output_dir)}')


    if args.pretrained_weight:
        model = SegNet.load_from_checkpoint(args.pretrained_weight)
    else:
        backbone_args = dict(
            spatial_dims=3,    
            in_channels=1,
            out_channels=len(copick_pickable_objects),
            backbone='resnet34',
            pretrained=False
        )
        model = SegNet(        
            nclasses = len(copick_pickable_objects),
            class_loss_weights = {p.name:p.metadata['class_loss_weight'] for p in copick_pickable_objects},
            backbone_args = backbone_args,
            lvl_weights = np.array([0, 0, 1, 1]),
            particle_ids = {p.name:i for i,p in enumerate(copick_pickable_objects)},
            particle_radius = {p.name:p.radius for p in copick_pickable_objects},
            particle_weights = {p.name:p.metadata['score_weight'] for p in copick_pickable_objects},
            score_thresholds = {p.name:p.metadata['score_threshold'] for p in copick_pickable_objects},
            output_dir = f'{args.output_dir}/jobs/{args.job_id}'
        )

    # Define Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        mode="max",
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
        logger=logger,
        num_sanity_val_steps=0,
    )

    # Train Model
    trainer.fit(model, data_module)



