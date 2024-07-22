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

#### Model pipeline

Prefect [flow](pipelines/flows/model_pipeline.py) that includes:
- [Data collection](pipelines/tasks/data_collection.py)
- [Data processing](pipelines/tasks/data_processing.py)
- [Model training](pipelines/tasks/model_training.py)
- [Model evaluation](pipelines/tasks/model_evaluation.py)

#### Models trained

For each model, a training Prefect flow or pipeline is built to train the model in a simple way. Additionally, a notebook is provided for experimentation.

- Baseline CNN
    * Trained from scratch
    * [Training pipeline](pipelines/models/baseline.py)
    * [Training notebook](model_notebooks/baseline.ipynb)
- ResNet-18
    * Fine-tuned
    * [Training pipeline](pipelines/models/resnet18.py)
    * [Training notebook](model_notebooks/resnet18.ipynb) 
- ResNet-34
    * Fine-tuned
    * [Training pipeline](pipelines/models/resnet34.py)
    * [Training notebook](model_notebooks/resnet34.ipynb)
- ResNet-50
    * Fine-tuned
    * [Training pipeline](pipelines/models/resnet50.py)
    * [Training notebook](model_notebooks/resnet50.ipynb)
- EfficientNet-B1
    * Fine-tuned
    * Used in app, best performing on mlflow experiments was taken
    * [Training pipeline](pipelines/models/efficientnet_b1.py)
    * [Training notebook](model_notebooks/efficientnet_b1.ipynb)
- DenseNet-121
    * Fine-tuned
    * [Training pipeline](pipelines/models/densenet121.py)
    * [Training notebook](model_notebooks/densenet121.ipynb)

#### Demo application

- [Inference module](app/inference.py)
    * Performs real-time inference of best performing model (ResNet-50)
    * Input: one image
    * Output: prediction, binary and logit
- [Flask server](app/app.py)
    * Provides an [API](app/api.py) for real-time inference using the [inference module](app/inference.py)
    * Includes a demo [front-end template](app/templates/index.html) to test the model, supporting image upload and prediction
- [Build action](.github/workflows/build.yml)
    * GitHub Action for building the web application
    * Builds the docker image, using the [dockerfile](Dockerfile) and the [docker compose yml](docker-compose.build.yml)
    * Registers the image to [docker hub](https://hub.docker.com/repository/docker/hyusta/brain-tumor-detection/general)

### Instructions
----------------------------------

#### Local development

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

Optionally, in order to see the prefect runs, run the prefect server
```{bash}
prefect server start
```

You are set! You can run notebooks, for example training [ResNet-18](models/resnet18.ipynb), [baseline CNN](models/baseline.ipynb) or [EDA](eda/eda.ipynb), or run some of the following
training pipelines:

```{bash}
python -m pipelines.models.baseline
```

```{bash}
python -m pipelines.models.resnet18
```

```{bash}
python -m pipelines.models.resnet34
```

```{bash}
python -m pipelines.models.resnet50
```

```{bash}
python -m pipelines.models.efficientnet_b1
```

```{bash}
python -m pipelines.models.densenet121
```

It is also possible to run the application in docker using:

```
docker compose up
```

#### Deploying the application

Find the latest tag to deploy (or alternatively find a specific tag to deploy)

```{bash}
export DOCKER_TAG=$(curl --silent "https://api.github.com/repos/humbertoyusta/brain-tumor-detection/tags" | grep '"name":' | sed -E 's/.*"([^"]+)".*/\1/' | head -n 1)
```

Fetch docker compose file

```{bash}
wget -O docker-compose.prod.yml https://raw.githubusercontent.com/humbertoyusta/brain-tumor-detection/$DOCKER_TAG/docker-compose.prod.yml
```

Spin up the application container

```{bash}
docker compose -f docker-compose.prod.yml up -d
```

You are set!, the application should be running on `localhost:5000`