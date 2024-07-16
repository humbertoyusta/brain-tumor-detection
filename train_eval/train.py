import os
import torch
import mlflow
import tqdm
import train_eval.loops
import utils.plots
import utils.mlflow


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        train_loader,
        val_loader,
        test_loader,
        num_epochs,
        model_name,
        mlflow_logging=False,
        train_transform=None,
        test_transform=None,
        plot=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.mlflow_logging = mlflow_logging
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.plot = plot

    def log_initial_params(self):
        mlflow.set_tag("model", self.model_name)

        mlflow.log_param("Number of Epochs", self.num_epochs)
        mlflow.log_param("Batch Size", self.train_loader.batch_size)

        mlflow.log_param("Train Dataset Size", len(self.train_loader.dataset))
        mlflow.log_param("Validation Dataset Size", len(self.val_loader.dataset))
        mlflow.log_param("Test Dataset Size", len(self.test_loader.dataset))

        mlflow.log_param("Model Name", self.model_name)
        mlflow.log_param(
            "Model Trainable Parameters",
            "{:,}".format(sum(p.numel() for p in self.model.parameters())),
        )

        mlflow.log_param("Model Architecture", self.model)

        mlflow.log_param("Optimizer", self.optimizer.__class__.__name__)
        mlflow.log_param(
            "Optimizer parameters",
            {
                "lr": self.optimizer.param_groups[0]["lr"],
                "betas": self.optimizer.param_groups[0]["betas"],
                "eps": self.optimizer.param_groups[0]["eps"],
                "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                "amsgrad": self.optimizer.param_groups[0]["amsgrad"],
            },
        )

        mlflow.log_param("Loss Function", self.criterion)

        mlflow.log_param("Train Transforms", self.train_transform)
        mlflow.log_param("Test Transforms", self.test_transform)

    def train(self):
        os.makedirs("checkpoints", exist_ok=True)

        train_losses = []
        val_losses = []
        val_accs = []
        best_val_loss = float("inf")

        with tqdm.tqdm(range(self.num_epochs)) as progress_bar:
            for epoch in progress_bar:
                train_loss = train_eval.loops.train_epoch(
                    self.model,
                    self.train_loader,
                    self.criterion,
                    self.optimizer,
                    self.device,
                )
                val_loss, val_acc, _ = train_eval.loops.eval_model(
                    self.model, self.val_loader, self.criterion, self.device
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        self.model.state_dict(),
                        os.path.join("checkpoints", f"{self.model_name}.pth"),
                    )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                if self.mlflow_logging:
                    mlflow.log_metric("Train Loss", train_loss, epoch)
                    mlflow.log_metric("Validation Loss", val_loss, epoch)
                    mlflow.log_metric("Validation Accuracy", val_acc, epoch)

                progress_bar.set_postfix(
                    {
                        "Train Loss": train_loss,
                        "Validation Loss": val_loss,
                        "Validation Accuracy": val_acc,
                    }
                )

        if self.mlflow_logging:
            mlflow.log_metric("Best Validation Loss", best_val_loss)
        if self.plot:
            utils.plots.plot_train_and_val_losses(train_losses, val_losses, val_accs)

    def get_final_model(self):
        self.model.load_state_dict(
            torch.load(os.path.join("checkpoints", f"{self.model_name}.pth"))
        )
        if self.mlflow_logging:
            mlflow.pytorch.log_model(self.model, "models")
        return self.model

    def run(self):
        if self.mlflow_logging:
            self.log_initial_params()

        self.train()
        return self.get_final_model()
