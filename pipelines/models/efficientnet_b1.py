import prefect
import torch
import torch.nn as nn
import torchvision
from typing import Tuple
import pipelines.flows.model_pipeline

train_model_name = "efficientnet_b1"


@prefect.task
def get_model(device: torch.device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torchvision.models.efficientnet_b1(
        weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT
    )
    model.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    return model, optimizer


@prefect.task
def get_epoch_finished_callback():
    def epoch_finished_callback(epoch_number: int, model: nn.Module):
        if epoch_number == 40:
            for name, param in model.named_parameters():
                if "classifier" not in name and "features.8" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        if epoch_number == 80:
            for name, param in model.named_parameters():
                if (
                    "classifier" not in name
                    and "features.8" not in name
                    and "features.7" not in name
                ):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    return epoch_finished_callback


@prefect.task
def get_extra_mlflow_params():
    return [
        ("Training Type", "Fine-Tuning"),
        (
            "Freezing Layers",
            {
                "Epochs 0-40": "All layers frozen except classifier",
                "Epochs 41-80": "All layers frozen except classifier and denseblock4.denselayer16",
                "Epochs 81-": "All layers frozen except classifier, denseblock4.denselayer15 and denseblock4.denselayer16",
            },
        ),
    ]


if __name__ == "__main__":
    pipelines.flows.model_pipeline.model_pipeline_flow(
        train_model_name=train_model_name,
        num_epochs=120,
        get_model=get_model,
        get_epoch_finished_callback=get_epoch_finished_callback,
        get_extra_mlflow_params=get_extra_mlflow_params,
    )
