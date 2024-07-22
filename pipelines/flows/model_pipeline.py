import torch.nn as nn
import pandas as pd
import prefect
import pipelines.tasks.data_collection
import pipelines.tasks.data_processing
import pipelines.tasks.model_training
import pipelines.tasks.model_evaluation
import pipelines.tasks.device


def generate_flow_run_name():
    time_now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"{time_now}"


@prefect.flow(
    name="Brain Tumor Detection - Model Pipeline", flow_run_name=generate_flow_run_name
)
def model_pipeline_flow(
    train_model_name: str,
    num_epochs: int,
    get_model: prefect.Task,
    get_epoch_finished_callback: prefect.Task,
    get_extra_mlflow_params: prefect.Task,
):
    pipelines.tasks.data_collection.collect_data()

    train_loader, val_loader, test_loader, train_transform, test_transform = (
        pipelines.tasks.data_processing.process_data()
    )

    device = pipelines.tasks.device.get_device()

    model, optimizer = get_model(device)

    model = pipelines.tasks.model_training.train_model(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_model_name=train_model_name,
        num_epochs=num_epochs,
        mlflow_logging=True,
        criterion=nn.BCELoss(),
        train_transform=train_transform,
        test_transform=test_transform,
        epoch_finished_callback=get_epoch_finished_callback(),
        extra_mlflow_params=get_extra_mlflow_params(),
    )

    pipelines.tasks.model_evaluation.evaluate_model(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.BCELoss(),
        device=device,
        mlflow_logging=True,
    )
