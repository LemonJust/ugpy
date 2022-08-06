"""
Structure from https://github.com/Lightning-AI/lightning/issues/9252
"""
import torch
from torch import nn
from torchmetrics import MetricCollection, F1Score, Precision, Recall, MeanMetric, Metric

from pytorch_lightning import LightningModule
from models import TwoBranchConv2d


class ImClassifier(LightningModule):
    """
    Implements the training and getting necessary metrics for the binary classification using imbalanced data.
    The loss is set to BCEWithLogitsLoss, the model can vary. Add the model to the choose_model()
    if you are adding to the list of the models that can work with this class.
    """

    def __init__(
            self,
            config
    ):
        """
        :param config: dictionary specifying the following :
            model: dict specifying what model to use. See models.py for a list of implemented models.
            loss: dict specifying what loss to use to be used in the loss.
            optimizer: dict specifying what optimiser to use .
        """
        super().__init__()

        self.config = config

        # set up model
        self.model = None
        self.criterion = None

        self.configure_model()
        self.configure_criterion()

        # set up metrics
        self.val_avg_loss = None
        self.val_metrics = None
        self.train_avg_loss = None
        self.train_metrics = None

        self.configure_metrics()

    def configure_model(self):
        if self.config["model"] == "TwoBranchConv2d":
            self.model = TwoBranchConv2d()
        else:
            # add to the list of available names if you implement another one
            raise ValueError('config["model"] can be only "TwoBranchConv2d"')

    def configure_criterion(self):

        if self.config["loss"] == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=
                                                  torch.tensor(self.config["loss_weight"]), reduction='mean')
        else:
            # add to the list of available names if you implement another one
            raise ValueError('config["loss"] can be only "BCEWithLogitsLoss"')

    def configure_metrics(self):
        # there are other ways for metrics to be identified as child modules:
        # https://torchmetrics.readthedocs.io/en/stable/pages/overview.html
        metrics = MetricCollection([F1Score(threshold=0.5),
                                    Precision(threshold=0.5),
                                    Recall(threshold=0.5)])
        self.train_metrics = metrics
        self.train_avg_loss = MeanMetric()

        # val and test use the same metrics, so only define validation
        self.val_metrics = metrics.clone()
        self.val_avg_loss = MeanMetric()

    def configure_optimizers(self):

        if self.config["optimizer"] == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.config["optimizer_lr"])
        else:
            # add to the list of available names if you implement another one
            raise ValueError('config["optimizer"] can be only "Adam"')

    @staticmethod
    def namespaced(name: str, metrics: MetricCollection):
        """
        Creates a description for logging the metrics correctly
        """
        return {f"{name}_{k}": v for k, v in metrics.items()}

    def step(self, batch, batch_idx, metrics):
        """
        One model step, this is the same for the train , validation and test
        Currently metrics are not used, but can use them to log something at a step.
        """
        x, y = batch
        outs = self.model(x)
        loss = self.criterion(outs, y)
        prob = torch.sigmoid(outs)

        return {"loss": loss, "prob": prob.detach(), "target": y.detach().int(), "n_samples": len(batch)}

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        outs = self.model(x)
        prob = torch.sigmoid(outs)
        return prob.type(torch.float16)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.train_metrics)

    def training_step_end(self, outputs):
        self.train_metrics(outputs["prob"], outputs["target"])
        self.train_avg_loss.update(outputs["loss"].repeat(outputs["n_samples"]))

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.val_metrics)

    def validation_step_end(self, outputs):
        self.val_metrics(outputs["prob"], outputs["target"])
        self.val_avg_loss.update(outputs["loss"].repeat(outputs["n_samples"]))

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, self.val_metrics)

    def test_step_end(self, outputs):
        self.val_metrics(outputs["prob"], outputs["target"])
        self.val_avg_loss.update(outputs["loss"].repeat(outputs["n_samples"]))

    def default_on_epoch_end(self, namespace: str, metrics: MetricCollection, avg_loss: Metric):
        computed = metrics.compute()
        computed = {"loss": avg_loss.compute(), **computed}
        computed = self.namespaced(f"{namespace}_avg", computed)
        self.log_dict(computed)

        metrics.reset()
        avg_loss.reset()

    def on_train_epoch_end(self) -> None:
        self.default_on_epoch_end("train", self.train_metrics, self.train_avg_loss)

    def on_validation_epoch_end(self) -> None:
        # sync_dist =True
        # https://forums.pytorchlightning.ai/t/synchronize-train-logging/1270
        self.default_on_epoch_end("val", self.val_metrics, self.val_avg_loss)

    def on_test_epoch_end(self) -> None:
        # sync_dist =True
        # https://forums.pytorchlightning.ai/t/synchronize-train-logging/1270
        self.default_on_epoch_end("test", self.val_metrics, self.val_avg_loss)


# if __name__ == "__main__":

