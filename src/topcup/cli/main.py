import click
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import monai
from torch.utils.data import DataLoader
from topcup.model import SegNet
from topcup.data.utils import worker_init_fn, collate_fn, train_collate_fn
from copick.impl.filesystem import CopickRootFSSpec
from topcup.data.copick_dataset import CopickDataset, TrainDataset
from topcup.data.augmentation import train_aug, get_basic_transform_list



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
            num_workers=4,  # should not be too many
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
            batch_size=self.batch_size, 
            num_workers=4, 
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last= True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return val_dataloader



class InferenceDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            copick_root = None,
            run_names = None,
            pixelsize = 10.012,
            recon_type = 'denoised',
            user_id = 'curation',
            batch_size: int = 1,
            has_ground_truth: bool=False,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.copick_root = copick_root
        self.run_names = run_names
        self.pixelsize = pixelsize
        self.recon_type= recon_type
        self.user_id = user_id
        self.has_ground_truth = has_ground_truth

    def setup(self, stage=None):    # stage='train' only gets called for training
        self.predict_dataset = CopickDataset(
            copick_root = self.copick_root,
            run_names = self.run_names,
            transforms = monai.transforms.Compose(get_basic_transform_list(["input"])),    
            pixelsize = self.pixelsize,
            recon_type = self.recon_type,
            user_id = self.user_id,
            has_ground_truth = self.has_ground_truth
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
            collate_fn=collate_fn,
            drop_last= True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            persistent_workers=True,
        )
        return predict_dataloader



################ sub commands ################
@click.command(name="train")
@click.option(
    "-c", "--copick_config",
    type=click.Path(exists=True, dir_okay=False, path_type=str), 
    help="copick config file path"
)
@click.option(
    "-tts", "--train_run_names", 
    type=str, 
    default="", 
    help="Tomogram dataset run names for training",
)
@click.option(
    "-vts", "--val_run_names", 
    type=str, 
    default="", 
    help="Tomogram dataset run names for validation"
)
@click.option(
    "-rt", "--reconstruction_type", 
    type=str, 
    default="denoised", 
    help="Tomogram reconstruction type. Default is denoised."
)
@click.option(
    "-u", "--user_id", 
    type=str, 
    default="curation", 
    help="Needed for training, the user_id used for the ground truth picks."
)
@click.option(
    "-s", "--session_id", 
    type=str, 
    default="None", 
    help="Needed for training, the session_id used for the ground truth picks. Default is None."
)
@click.option(
    "-bs", "--batch_size", 
    type=int, 
    default=8, 
    help="batch size for data loader"
)
@click.option(
    "-n", "--n_aug", 
    type=int, 
    default=1112, 
    help="Data augmentation copy. Default is 1112."
)
@click.option(
    "-l", "--learning_rate", 
    type=float, 
    default=1e-4, 
    help="Learning rate for optimizer"
)
@click.option(
    "-p", "--pretrained_weight", 
    type=str, 
    default="", 
    help="One pretrained weights file path. Default is None."
)
@click.option(
    "-e", "--epochs", 
    type=int, 
    default=100, 
    help="Number of epochs. Default is 100."
)
@click.option(
    "--pixelsize", 
    type=float,
    default=10.0, help="Pixelsize in angstrom. Default is 10.0A."
)
@click.option(
    "-o", "--output_dir", 
    type=str, 
    default="./output", 
    help="output dir for saving checkpoints"
)
@click.option(
    "-j", "--job_id", 
    type=str, 
    default="job_0", 
    help="Unique id for each run"
)
@click.pass_context
def train(
    ctx,
    copick_config,
    train_run_names,
    val_run_names,
    reconstruction_type,
    user_id,
    session_id,
    batch_size,
    n_aug,
    learning_rate,
    pretrained_weight,
    epochs,
    pixelsize,
    output_dir,
    job_id
):
    copick_root = CopickRootFSSpec.from_file(copick_config)
    copick_pickable_objects = []
    for obj in copick_root.pickable_objects:
        if obj.is_particle:
            if obj.radius:
                copick_pickable_objects.append(obj)
            else:
                print(f'Skipping {obj.name} in data loading becuase it does not have a radius in the copick configuration file.')
    
    data_module = DataModule(
        copick_root=copick_root, 
        train_run_names=train_run_names.split(','), 
        val_run_names=val_run_names.split(','), 
        batch_size=batch_size, 
        pixelsize=pixelsize,
        recon_type=reconstruction_type,
        user_id=user_id,
        session_id=session_id,
        n_aug=n_aug
    )
    output_dir = Path(f'{output_dir}/jobs/{job_id}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'making output dir {str(output_dir)}')


    if pretrained_weight:
        model = SegNet.load_from_checkpoint(pretrained_weight)
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
            learning_rate = learning_rate,
            particle_ids = {p.name:i for i,p in enumerate(copick_pickable_objects)},
            particle_radius = {p.name:p.radius for p in copick_pickable_objects},
            particle_weights = {p.name:p.metadata['score_weight'] for p in copick_pickable_objects},
            score_thresholds = {p.name:p.metadata['score_threshold'] for p in copick_pickable_objects},
            output_dir = f'{output_dir}/jobs/{job_id}'
        )

    # Define Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",
        mode="max",
        save_top_k=1,
        dirpath=f"{output_dir}/checkpoints/",
        filename="best_model"
    )

    # Logger
    logger = CSVLogger(save_dir=f"{output_dir}/logs/", name="training_logs")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=2,
        callbacks=[checkpoint_callback],
        logger=logger,
        num_sanity_val_steps=0,
    )

    # Train Model
    trainer.fit(model, data_module)




@click.command(name="inference")
@click.option(
    "-c", "--copick_config",
    type=click.Path(exists=True, dir_okay=False, path_type=str),  
    help="copick config file path")
@click.option(
    "-ts", "--run_names", 
    type=str, 
    default="", 
    help="Tomogram dataset run names"
)
@click.option(
    "-bs", "--batch_size", 
    type=int, 
    default=1, 
    help="batch size for data loader"
)
@click.option(
    "-p", "--pretrained_weights", 
    type=str, 
    default="", 
    help="Pretrained weights file paths (use comma for multiple paths). Default is None."
)
@click.option(
    "-pa", "--pattern", 
    type=str, 
    default="*.ckpt", 
    help="The key for pattern matching checkpoints. Default is *.ckpt"
)
@click.option(
    "--pixelsize", 
    type=float, 
    default=10.0, 
    help="Pixelsize in angstrom. Default is 10.0A."
)
@click.option(
    "-rt", "--reconstruction_type", 
    type=str, 
    default="denoised", 
    help="Tomogram reconstruction type. Default is denoised."
)
@click.option(
    "-u", "--user_id", 
    type=str, 
    default="curation", 
    help="Needed for training, the user_id used for the ground truth picks."
)
@click.option(
    "-o", "--output_dir", 
    type=str, 
    default="./output", 
    help="output dir for saving prediction results (csv)."
)
@click.option(
    "-g", "--gpus", 
    type=int, 
    default=1, 
    help="Number of GPUs for inference. Default is 1."
)
@click.option(
    "-gt", "--has_ground_truth", 
    type=bool, 
    default=False, 
    help="Inference with ground truth annoatations"
)
@click.pass_context
def inference(
    ctx,
    copick_config,
    run_names,
    batch_size,
    pretrained_weights,
    pattern,
    pixelsize,
    reconstruction_type,
    user_id,
    output_dir,
    gpus,
    has_ground_truth
):
    copick_root = CopickRootFSSpec.from_file(copick_config)
    data_module = InferenceDataModule(
        copick_root=copick_root, 
        run_names=run_names.split(','), 
        batch_size=batch_size, 
        pixelsize=pixelsize, 
        recon_type=reconstruction_type,
        user_id = user_id, 
        has_ground_truth=has_ground_truth
    )

    output_dir = Path(f'{output_dir}')
    print(f'making output dir {str(output_dir)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models from checkpoints
    ensemble_model = SegNet.ensemble_from_checkpoints(pretrained_weights, pattern=pattern)
    ensemble_model.output_dir = output_dir
    ensemble_model.score_thresholds = {p.name:p.metadata['score_threshold'] for p in copick_root.pickable_objects}

    # Initialize trainer
    trainer = pl.Trainer(devices=1, accelerator="gpu") if gpus == 1 else pl.Trainer(devices=gpus, accelerator="gpu", strategy="ddp")
    trainer.predict(ensemble_model, datamodule=data_module)








