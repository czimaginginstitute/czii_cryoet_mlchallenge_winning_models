# CZII_ML_Challenge_Winning_Models
The re-fractory codes of 1st winning team's solution [kaggle-cryoet-1st-place-segmentation](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation/tree/main).

# Installation
cd into the root folder, then 
```
pip install -r requirements.txt
pip install -e .
```

or run the enviroment in the container as suggested by the original repository. Below is an example of using apptainer on CZBiohub HPC.

```
apptainer pull pytorch_24.08.sif docker://nvcr.io/nvidia/pytorch:24.08-py3
apptainer run --bind VOLUME_TO_MOUNT --nv pytorch_24.08.sif
```


# Training 
Traning dataset (localizations) is hard coded file train_folded_v1.csv, need to put in the current working dir. Command example for training: 
```
python train.py -c cfg_resnet34 -i /hpc/projects/group.czii/kevin.zhao/ml_challenge/winning_models/data/train/static/ExperimentRuns/ -o checkpoint --fold -1
```


# Inference
Example command for inference with PyTorch checkpoints: 

```
python inference.py -c cfg_resnet34 -d 0 -i /hpc/projects/group.czii/kevin.zhao/ml_challenge/winning_models/data/test/ -p checkpoints/fold-1/
```