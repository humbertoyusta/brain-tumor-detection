import prefect
import torch
import torch.nn
import torch.utils.data
import mlflow
import train_eval.eval


@prefect.task
def evaluate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    mlflow_logging: bool = True,
) -> None:

    evaluator = train_eval.eval.ModelEvaluator(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        mlflow_logging=mlflow_logging,
        plot=False,
    )
    evaluator.run()

    if mlflow_logging:
        mlflow.end_run()
