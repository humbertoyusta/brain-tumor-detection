import prefect
import torch
import torch.nn as nn
from typing import Tuple
import pipelines.flows.model_pipeline

train_model_name = "baseline_cnn"


@prefect.task
def get_model(device: torch.device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 12 * 12, 128),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer


@prefect.task
def get_epoch_finished_callback():
    def epoch_finished_callback(epoch_number: int, model: nn.Module):
        return

    return epoch_finished_callback


@prefect.task
def get_extra_mlflow_params():
    return [
        ("Training Type", "Train From Scratch"),
    ]


if __name__ == "__main__":
    pipelines.flows.model_pipeline.model_pipeline_flow(
        train_model_name=train_model_name,
        num_epochs=90,
        get_model=get_model,
        get_epoch_finished_callback=get_epoch_finished_callback,
        get_extra_mlflow_params=get_extra_mlflow_params,
    )
