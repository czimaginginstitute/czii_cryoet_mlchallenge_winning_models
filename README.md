# CZII_ML_Challenge_Winning_Models
The re-implementation of 1st winning team's solution [kaggle-cryoet-1st-place-segmentation](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation/tree/main) in pytorch-lightning and copick..

# Installation
cd into the root folder, then 
```
pip install -r requirements.txt
pip install -e .
```


# Training 
Traning dataset (localizations) is hard coded file train_folded_v1.csv, need to put in the current working dir. Command example for training: 
```
python train.py -c COPICK_CONFIG_FILE -o checkpoint 
```


# Inference
Example command for inference with PyTorch checkpoints: 

```
python inference.py -c COPICK_CONFIG_FILE -p checkpoints/fold-1/
```