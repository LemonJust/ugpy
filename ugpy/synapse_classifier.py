"""
Structure from https://github.com/Lightning-AI/lightning/issues/9252
"""
import glob
import os

import numpy as np
import tifffile as tif

import torch
from torch import nn
from torchmetrics import MetricCollection, F1Score, Precision, Recall, MeanMetric, Metric

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from datasets import TwoSlicesDataModule, TwoSlicesProbMapModule
from models import TwoBranchConv2d
from preprocess import split_to_rois, get_image_shape, Image
from loader import load_image


class SynapseClassifier(LightningModule):
    """
    Implements the training and logging of necessary metrics for the imbalanced data classification,
    as is the case with the synapse classification.
    """

    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module = None,
            lr: float = 0.0002
    ):
        """
        :param model: model to train. See models.py for a list of implemented models.
        :param criterion: loss to use with the provided model.
        :param lr: learning rate to use with the optimiser.
        """
        super().__init__()

        self.model = model
        self.criterion = criterion

        # valid ways metrics will be identified as child modules:
        # https://torchmetrics.readthedocs.io/en/stable/pages/overview.html
        metrics = MetricCollection([F1Score(threshold=0.5),
                                    Precision(threshold=0.5),
                                    Recall(threshold=0.5)])
        self.train_metrics = metrics
        self.train_avg_loss = MeanMetric()

        # val and test use the same metrics, so only define validation
        self.val_metrics = metrics.clone()
        self.val_avg_loss = MeanMetric()

        self.lr = lr
        self.save_hyperparameters(ignore=['model', 'criterion'])

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def predict_prob_map_with_logger(image_file, image_name, run_id, rois=None):
    """
    Creates a probability map , working in chunks. Chunks are currently set up to work on my machine,
    but might need to be entered by user in the future.
    """
    wandb_logger = WandbLogger(project=project_name, job_type='prob_map')

    # you might want to change it for different machines
    num_workers = 0

    # # TODO : try wandb restore
    checkpoint_path = glob.glob(f"D:/Code/repos/UGPy/ugpy/wandb/*{run_id}/files/*.ckpt")[0]
    print(f"Using checkpoint : {checkpoint_path}")
    model = SynapseClassifier.load_from_checkpoint(checkpoint_path)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=1, accelerator='gpu', devices=1, logger=wandb_logger)

    if rois is None:
        # use chunks to draw probability map in pieces
        zyx_chunks = [5, 5, 5]
        zyx_padding = [10, 10, 10]
        rois = split_to_rois(image_file, zyx_chunks, zyx_padding)

    prob_map = np.zeros(get_image_shape(image_file), dtype=np.int16)
    img = load_image(image_file)
    for roi in rois:
        data_module = TwoSlicesProbMapModule([roi], img=img, num_workers=num_workers)
        prediction = trainer.predict(model, datamodule=data_module, ckpt_path=checkpoint_path)[0]
        print(len(prediction))
        print(prediction.requires_grad)
        print(prediction.device)

        SCALE = 1000
        for i, pixel in enumerate(data_module.pred_dataset.centroids):
            prob_map[pixel[0], pixel[1], pixel[2]] = int(prediction[i] * SCALE)
        break

    RESOLUTION = [0.68, 0.23, 0.23]
    prob_image = Image(RESOLUTION, img=prob_map.astype(np.int16))
    save_dir = "D:/Code/repos/UGPy/data/predict/prob_map/maps"
    save_name = f"map_{image_name}_run_id{run_id}.tif"
    prob_image.imwrite(os.path.join(save_dir, save_name))

    wandb.config["image"] = image_name
    wandb.config["run_id"] = run_id
    wandb.config["num_workers"] = num_workers

    # close wandb run
    wandb.finish()


def train_and_test_with_logger(project_name, seed, max_epochs, pos_weight, batch_size, num_workers=1):
    deterministic = True
    if deterministic:
        # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
        pl.seed_everything(seed, workers=True)

    # Initialize wandb logger
    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.wandb.html
    wandb_logger = WandbLogger(project=project_name, job_type='train', log_model=True)  # name="test1"

    data_module = TwoSlicesDataModule(batch_size=batch_size,
                                      num_workers=num_workers)

    # set up the model
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='mean')
    model = SynapseClassifier(TwoBranchConv2d(), criterion, lr=0.0002)

    # remember the model with the best validation loss
    checkpoint_callback = ModelCheckpoint(monitor='val_avg_loss', mode="min",
                                          dirpath=wandb.run.dir,
                                          filename="checkpoint_best_{val_avg_loss:.4f}_{epoch}")
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator='gpu', devices=1,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         deterministic=deterministic
                         )

    # Train the model
    trainer.fit(model, data_module)

    # Evaluate the model on the held-out test set
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)

    # this should work, but doesn't :
    # TODO : should I report that id doesn't work?
    # https://pytorch-lightning.readthedocs.io/en/stable/common/evaluation.html#test-after-fit
    # (2) test using a specific checkpoint
    # trainer.test(ckpt_path="/path/to/my_checkpoint.ckpt")

    # add some more info to the log:
    # TODO: use wandb config to set up the experiment

    wandb.config["loss"] = "BCEWithLogitsLoss"
    wandb.config["pos_weight"] = pos_weight

    wandb.config["model_name"] = "TwoBranchConv2d"
    wandb.config["input_shape"] = "1x15x15"
    wandb.config["normalization"] = "standardise by fish"
    wandb.config["batch_size"] = batch_size

    wandb.config["optimiser"] = "Adam"

    wandb.config["train_roi"] = data_module.training_roi_ids
    wandb.config["test_roi"] = data_module.testing_roi_ids

    wandb.config["num_train"] = len(data_module.train_dataset)
    wandb.config["num_val"] = len(data_module.val_dataset)
    wandb.config["num_test"] = len(data_module.test_dataset)

    wandb.config["frac1_train"] = data_module.frac1_train
    wandb.config["frac1_val"] = data_module.frac1_val
    wandb.config["frac1_test"] = data_module.frac1_test

    wandb.config["drop_unsegmented"] = "all"
    wandb.config["seed"] = seed
    wandb.config["deterministic"] = deterministic

    # close wandb run
    wandb.finish()


def predict(project_name):
    wandb_logger = WandbLogger(project=project_name, job_type='predict')

    img_file = "D:/Code/repos/UGPy/data/predict/prob_map/1-20FJ.tif"
    # use chunks to draw probability map in pieces
    zyx_chunks = [3, 3, 3]
    zyx_padding = [10, 10, 10]
    rois = split_to_rois(img_file, zyx_chunks, zyx_padding)
    num_workers = 0
    batch_size = 50000
    data_module = TwoSlicesProbMapModule(img_file, [rois[0]], batch_size, num_workers=num_workers)

    # set up the model
    # # TODO : make it so that I don't need to set up the loss ?
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5), reduction='mean')
    model = SynapseClassifier(TwoBranchConv2d(), criterion, lr=0.0002)

    # # get the check point
    # # TODO : try wandb restore
    run_id = "3m51q4ne"
    checkpoint_path = glob.glob("D:/Code/repos/UGPy/ugpy/wandb/*3m51q4ne/files/*.ckpt")[0]
    # glob.glob("D:/Code/repos/UGPy/ugpy/wandb/*3m51q4ne/files/*.ckpt")
    print(f"Using checkpoint : {checkpoint_path}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=1,
                         accelerator='gpu', devices=1,
                         logger=wandb_logger)
    prediction = trainer.predict(model, datamodule=data_module, ckpt_path=checkpoint_path)

    wandb.config["batch_size"] = batch_size
    wandb.config["num_pred"] = len(data_module.pred_dataset)
    wandb.config["num_workers"] = num_workers

    # close wandb run
    wandb.finish()


if __name__ == "__main__":
    seed = 222
    max_epochs = 20
    pos_weight = 5
    batch_size = 500
    project_name = 'UGPy-SynapseClassifier'
    train_and_test_with_logger(project_name, seed, max_epochs, pos_weight, batch_size)
    #
    # img_file = "D:/Code/repos/UGPy/data/predict/prob_map/raw/1-20FJ.tif"
    # run_id = "3m51q4ne"
    # predict_prob_map_with_logger(img_file, "1-20FJ", run_id, rois=None)
