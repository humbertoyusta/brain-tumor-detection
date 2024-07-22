import prefect
import torch
import torch.nn as nn
import torchvision
from typing import Tuple
import pipelines.flows.model_pipeline

train_model_name = "resnet34"


@prefect.task
def get_model(device: torch.device) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torchvision.models.resnet34(
        weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    )
    model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

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
                if "fc" not in name and "layer4" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        if epoch_number == 80:
            for name, param in model.named_parameters():
                if "fc" not in name and "layer4" not in name and "layer3" not in name:
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
                "Epochs 0-40": "All layers frozen except fc",
                "Epochs 41-80": "All layers frozen except fc and layer4",
                "Epochs 81-": "All layers frozen except fc, layer4 and layer3",
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
