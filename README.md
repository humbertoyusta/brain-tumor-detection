## Brain tumor detection

Main project for Industrial Machine Learning course at Harbour.Space University on July 2024.

[Slides about problem statement](https://docs.google.com/presentation/d/1wnG2xsezhYPLNx-wTspJytdLF2cgTPMhapRH0RburEE/edit?usp=sharing)

[Slides about experiments](https://docs.google.com/presentation/d/1q3enluvHo8W3Bu0Q_yl7RypwO3J-KBCnXjBT3jnZ4Ag/edit?usp=sharing)

### Project components
----------------------------------

[Exploratory data analysis](eda/eda.ipynb)
- Duplication examples visualization
- Visualizing class distribution
- Pixel intensity statistics
- Pixel correlation matrix 
- t-SNE visualization of the data
- Blur analysis using variance of Laplacian

#### Data handling
- [Data collector](preprocessing/data_collector.py)
- [Data processor](preprocessing/data_processor.py)
    * Check image corruption and assert image integrity
    * Deduplicate images using perceptual hashing with [imagededup](https://github.com/idealo/imagededup) library
    * Split dataset into train, validation and test
    * Define augmentations

#### Data handling visualization
- [Crop brain region visualization](preprocessing/crop_brain_region_visualization.ipynb), [crop brain region](preprocessing/crop_brain_region.py) is implemented as custom albumentation transform
- [Augmentations visualization](preprocessing/augmentations_visualization.ipynb), augmentations use [albumentations](https://albumentations.ai/) library
- Shapley values of baseline, last section of [this notebook](models/baseline.ipynb), using [shap](https://shap.readthedocs.io/en/latest/) library

#### Model training and evaluation
- [Model training](train_eval/train.py)
    * Log params, metrics and model artifact to mlflow
    * Pytorch training loop, saves model with best validation loss across all epochs
- [Model evaluation](train_eval/eval.py)
    * Log model metrics on test and validation datasets to mlflow
    * Optionally plot losses, confusion matrix and other metrics (accuracy, precision, recall, F1)

#### Models trained
- [Baseline CNN](models/baseline.ipynb)
    * Trained from scratch
- [ResNet-18](models/resnet18.ipynb)
    * Fine tuned 


### Instructions
----------------------------------

Clone repo and cd into it

```{bash}
git clone git@github.com:humbertoyusta/brain-tumor-detection.git
cd brain-tumor-detection
```

(Using SSH, can also be done using HTTPS)

Create activate virtual environment (recommended)

```{bash}
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```{bash}
pip install -r requirements.txt
```

To be able to collect data from kaggle, if you haven't done it yet, go to your [kaggle account settings](https://www.kaggle.com/settings), create a new token, and place it in `~/.kaggle/kaggle.json`

Start the mlflow tracking server (needed for training and evaluating models with logging to ML Flow)

```{bash}
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

You are set! You can run notebooks, for example training [ResNet-18](models/resnet18.ipynb), [baseline CNN](models/baseline.ipynb) or [EDA](eda/eda.ipynb)