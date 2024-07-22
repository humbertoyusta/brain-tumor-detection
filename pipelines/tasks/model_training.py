import prefect
import torch
import torch.nn
import torch.utils.data
import pandas as pd
import albumentations
import mlflow
from typing import Optional, Callable, Tuple, Any, List
import train_eval.train


@prefect.task
def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    train_model_name: str,
    num_epochs: int,
    mlflow_logging=True,
    criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    train_transform: Optional[albumentations.Compose] = None,
    test_transform: Optional[albumentations.Compose] = None,
    epoch_finished_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    extra_mlflow_params: List[Tuple[str, Any]] = [],
) -> torch.nn.Module:
    if mlflow_logging:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        mlflow.set_experiment("Brain Tumor Detection - Model Training and Evaluation")

        time_now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

        mlflow.start_run(
            run_name=f"{train_model_name}-{time_now}", log_system_metrics=True
        )

    trainer = train_eval.train.ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        model_name=train_model_name,
        mlflow_logging=mlflow_logging,
        train_transform=train_transform,
        test_transform=test_transform,
        plot=False,
        epoch_finished_callback=epoch_finished_callback,
        extra_mlflow_params=extra_mlflow_params,
    )

    return trainer.run()
