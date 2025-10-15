#  TopCUP: Top CryoET U-Net Picker
The re-implementation of 1st winning team's solution [kaggle-cryoet-1st-place-segmentation](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation/tree/main) in pytorch-lightning and copick.


## Performance
We are able to train 3 models (resnet34 backbones) with 6, 12, and 24 tomograms respectively, and achieved an esenmble score of 0.774. This is comparable to the original submission of the 1st place [kaggle-cryoet-leader-board](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/leaderboard).
<p align="center">
  <img src="assets/scores.png" alt="F4 score of each protein complex using different training set sizes">
</p>



## Installation
```
pip install git+https://github.com/czimaginginstitute/czii_cryoet_mlchallenge_winning_models.git
```

Or cd into the root folder, then 
```
pip install -e .
```

## Copick configuration file
The copick data ingestion can automatically populate many important internal variables from the config file. Especially, the metrics for the training and evaluation process, such as `class_loss_weight`, `score_threshold`, and `score_weight` are stored under the metadata key in the configuration file. 

- `class_loss_weight`: weighting each class in the DenseCrossEntrope loss
- `score_threshold`: white filter picks per class above the value from the final probability--reduce false positives 
- `score_weight`: weighting each class in the F beta score 

An example of copick config file is shown below:
```
{
    "name": "Phatom Dataset",
    "description": "CZII ML Challenge Training dataset",
    "version": "1.0.1",
    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 255],
            "radius": 60,
            "map_threshold": 0.0418,
            "metadata": {
                "score_weight": 1,
                "score_threshold": 0.16,
                "class_loss_weight": 256
            }
        },
        {
            "name": "beta-amylase",
            "is_particle": true,
            "pdb_id": "1FA2",
            "label": 2,
            "color": [153,  63,   0, 255],
            "radius": 65,
            "map_threshold": 0.035,
            "metadata": {
                "score_weight": 0,
                "score_threshold": 0.25,
                "class_loss_weight": 256
            }
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 255],
            "radius": 90,
            "map_threshold": 0.0578,
            "metadata": {
                "score_weight": 2,
                "score_threshold": 0.13,
                "class_loss_weight": 256
            }
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 255],
            "radius": 150,
            "map_threshold": 0.0374,
            "metadata": {
                "score_weight": 1,
                "score_threshold": 0.19,
                "class_loss_weight": 256
            }
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 255],
            "radius": 130,
            "map_threshold": 0.0278,
            "metadata": {
                "score_weight": 2,
                "score_threshold": 0.18,
                "class_loss_weight": 256
            }
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "label": 6,
            "color": [255, 204, 153, 255],
            "radius": 135,
            "map_threshold": 0.201,
            "metadata": {
                "score_weight": 1,
                "score_threshold": 0.5,
                "class_loss_weight": 256
            }
        }
    ],
    "config_type": "czii cryoet mlchallenge dataset",
    "overlay_root": "local:///PATH/TO/EXTRACTED/PROJECT/",
    "static_root": "local:///PATH/TO/EXTRACTED/PROJECT/"
}
```

## Commands
After installation, use the command `topcup --help` to show all the possible subcomamnds:
```
Usage: topcup [OPTIONS] COMMAND [ARGS]...

  topcup: a top crypet u-net picker

Options:
  -v, --verbose  Increase verbosity (-v, -vv).
  --version      Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  inference
  train
  score
```

### Training from scratch
Use `topcup train --help` to see all the options for training. The code support loading data via copick. An example training command is below.
```
topcup train \
    --copick_config COPICK_CONFIG_FILE \
    --train_run_names TS_6_4,TS_6_6,TS_69_2,TS_73_6,TS_86_3,TS_99_9 \
    --val_run_names TS_5_4 \
    --tomo_type denoised \
    --user_id COPICK_USER_ID \
    --pixelsize 10 \  
    --batch_size 16 \
    --n_aug 1112 \
    --output_dir OUTPUT_PATH \
    --logger_version 1 \
    --epochs 100   
```

### Re-training from a checkpoint for the same dataset
```
topcup train \
    --copick_config COPICK_CONFIG_FILE \
    --train_run_names TS_6_4,TS_6_6,TS_69_2,TS_73_6,TS_86_3,TS_99_9 \
    --val_run_names TS_5_4  \
    --tomo_type denoised \
    --user_id COPICK_USER_ID \
    --pixelsize 10 \  
    --batch_size 16 \
    --n_aug 1112 \
    --output_dir OUTPUT_PATH \
    --logger_version 1 \
    --epochs 100 \
    --pretrained_weight CHECKPOINT_PATH   
```

### *Subset transfer learning: re-training from a checkpoint for a different dataset 
**Subset transfer learning** involves loading a checkpoint from a pretrained model and fine-tuning it on a new dataset that includes only a subset of the original classes. To do this correctly, it’s important to know which classes and the corresponding `channel_id` the original model was trained on. This information can be accessed by loading the checkpoint and inspecting the `model.description` attribute. The `copick_config` used for fine-tuning should include the same pickable objects as the original training setup, with updated class weights and thresholds as needed for the new task.

```
>>> from czii_cryoet_models.model import SegNet
>>> model = SegNet.load_from_checkpoint('/hpc/projects/group.czii/kevin.zhao/ml_challenge/winning_models/czii_cryoet_mlchallenge_models/output_test/checkpoints/best_model-v6.ckpt')
>>> print(model.description)
SegNet model predicting 6 classes

Class details:
{
  "apo-ferritin": {
    "channel_id": 0,
    "radius": 60.0,
    "score_threshold": 0.16,
    "score_weight": 1
  },
  "beta-amylase": {
    "channel_id": 1,
    "radius": 65.0,
    "score_threshold": 0.25,
    "score_weight": 0
  },
  "beta-galactosidase": {
    "channel_id": 2,
    "radius": 90.0,
    "score_threshold": 0.13,
    "score_weight": 2
  },
  "ribosome": {
    "channel_id": 3,
    "radius": 150.0,
    "score_threshold": 0.19,
    "score_weight": 1
  },
  "thyroglobulin": {
    "channel_id": 4,
    "radius": 130.0,
    "score_threshold": 0.18,
    "score_weight": 2
  },
  "virus-like-particle": {
    "channel_id": 5,
    "radius": 135.0,
    "score_threshold": 0.5,
    "score_weight": 1
  }
}
```


### Inference
Use command `topcup inference --help` to see all the options for the inference pipeline. An example command for inference with PyTorch checkpoints (a single checkpoint file path or multiple folder paths, each containing mutiple checkpoints) that supports pattern matching. 

```
topcup inference \
    --copick_config copick_config.json \
    --run_names TS_100_4,TS_100_6,TS_100_7,TS_100_9 \
    --tomo_type denoised \
    --user_id COPICK_USER_ID \
    --pretrained_weights FOLDER_PATH1/checkpoints/,FOLDER_PATH2/checkpoints/,FOLDER_PATH3/checkpoints/ \
    --batch_size 16 \
    --output_dir OUTPUT_PATH \
    --pattern *v1.ckpt 
```

### Score calculation
Use command 'topcup score --help' to see all the options for calculating F-beta score for the predictions:
```
Usage: topcup score [OPTIONS]

Options:
  -c, --copick_config FILE  copick config file path  [required]
  -g, --gt FILE             Ground truth picks csv file path  [required]
  -s, --submission FILE     Submission picks csv file path  [required]
  -h, --help                Show this message and exit.
```


## 📚 Documentation

Coming soon.

## 🤝 Contributor covenant code of conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

Responsible Use: We are committed to advancing the responsible development and use of artificial intelligence. Please follow our [Acceptable Use Policy](https://virtualcellmodels.cziscience.com/acceptable-use-policy) when engaging with the model.

## 🔒 Security

If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.
