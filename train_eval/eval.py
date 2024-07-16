import utils.mlflow
import utils.plots
import train_eval.loops


class ModelEvaluator:
    def __init__(
        self,
        model,
        val_loader,
        test_loader,
        criterion,
        device,
        mlflow_logging=False,
        plot=False,
    ):
        self.model = model
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.mlflow_logging = mlflow_logging
        self.plot = plot

    def run(self):
        _, _, val_confusion_matrix = train_eval.loops.eval_model(
            self.model, self.val_loader, self.criterion, self.device
        )
        _, _, test_confusion_matrix = train_eval.loops.eval_model(
            self.model, self.test_loader, self.criterion, self.device
        )

        val_metrics = utils.metrics.get_metrics_from_confusion_matrix(
            val_confusion_matrix
        )
        test_metrics = utils.metrics.get_metrics_from_confusion_matrix(
            test_confusion_matrix
        )

        if self.mlflow_logging:
            utils.mlflow.log_evaluation_metrics(
                "Validation", val_metrics, val_confusion_matrix
            )
            utils.mlflow.log_evaluation_metrics(
                "Test", test_metrics, test_confusion_matrix
            )

        if self.plot:
            utils.plots.plot_evaluation_result(
                val_metrics, test_metrics, val_confusion_matrix, test_confusion_matrix
            )
