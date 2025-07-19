# CZII_ML_Challenge_Winning_Models
The re-implementation of 1st winning team's solution [kaggle-cryoet-1st-place-segmentation](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation/tree/main) in pytorch-lightning and copick..


## Benchmark
We are able to train 3 models (resnet34 backbones) with 6, 12, and 24 tomograms respectively, and achieved an esenmble score of 0.774. This is comparable to the original submission of the 1st place [kaggle-cryoet-leader-board](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/leaderboard).
<p align="center">
  <img src="assets/scores.png" alt="F4 score of each protein complex using different training set sizes">
</p>



## Installation
cd into the root folder, then 
```
pip install -e .
```


## Training from scratch
The code support loading data via copick and directly loading zarr data. An example training command is below.
```
python train.py \
    --copick_config COPICK_CONFIG_FILE \
    --train_run_names TS_6_4,TS_6_6,TS_69_2,TS_73_6,TS_86_3,TS_99_9 \
    --val_run_names TS_5_4 \
    --batch_size 16 \
    --n_aug 1112 \
    --output_dir OUTPUT_PATH \
    --job_id job_1 \
    --epochs 100   
```

## Re-training from a checkpoint
```
python train.py \
    --copick_config COPICK_CONFIG_FILE \
    --train_run_names TS_6_4,TS_6_6,TS_69_2,TS_73_6,TS_86_3,TS_99_9 \
    --val_run_names TS_5_4  \
    --batch_size 16 \
    --n_aug 1112 \
    --output_dir OUTPUT_PATH \
    --job_id job_1 \
    --epochs 100 \
    --pretrained_weight CHECKPOINT_PATH   
```


## Inference
An example command for inference with PyTorch checkpoints (a single checkpoint file path or multiple folder paths, each containing mutiple checkpoints) that supports pattern matching. 

```
python inference.py \
    --copick_config copick_config.json \
    --run_names TS_100_4,TS_100_6,TS_100_7,TS_100_9 \
    --pretrained_weights FOLDER_PATH1/checkpoints/,FOLDER_PATH2/checkpoints/,FOLDER_PATH3/checkpoints/ \
    --batch_size 16 \
    --output_dir OUTPUT_PATH \
    --pattern *v1.ckpt 
```

Inference by loading zarr files:
```
python inference_custom.py \
    --file_path FOLDER_PATH_TO_ZARR_FILES \
    --pretrained_weights oFOLDER_PATH1/checkpoints/,FOLDER_PATH2/checkpoints/,FOLDER_PATH3/checkpoints/ \
    --batch_size 16 \
    --output_dir OUTPUT_PATH \
    --pixelsize PIXEL_SIZE \
    --pattern *v1.ckpt 
```