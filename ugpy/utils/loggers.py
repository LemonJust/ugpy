from pytorch_lightning.callbacks import Callback
import torch
from torchvision.utils import make_grid
import wandb


class ImagePredictionLogger(Callback):
    """
    Sends examples of images and their labels to the w&b
    NOT finished ! needs a dataloader !
    """

    def __init__(self, dataloader):
        super().__init__()
        (self.imgs1, self.imgs2), self.labels = next(iter(dataloader))

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        imgs1 = self.imgs1.to(device=pl_module.device)
        imgs2 = self.imgs2.to(device=pl_module.device)
        labels = self.labels.to(device=pl_module.device)

        # Get model prediction ( need sigmoid since using BCEWithLogits )
        logits = torch.sigmoid(pl_module(imgs1, imgs2))
        threshold = torch.tensor([0.5])
        preds = (logits > threshold).float() * 1

        # Log the images as wandb Image
        # TODO: use wandb directly
        trainer.logger.experiment.log({
            "examples": [wandb.Image(make_grid([img1, img2]), caption=f"Pred:{pred}, Label:{y}")
                         for img1, img2, pred, y in zip(imgs1, imgs2, preds, labels)]
        })

# class ImageCMLogger(Callback):
#     """
#     Records some examples of images for each type in the confusion metrics : TP, FP, TN, FN
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def on_test_epoch_end(self, trainer, pl_module):
#         self.imgs1, self.imgs2, self.labels = next(iter(pl_module.example_dataloader))
#         # Bring the tensors to CPU
#         imgs1 = self.imgs1.to(device=pl_module.device)
#         imgs2 = self.imgs2.to(device=pl_module.device)
#         labels = self.labels.to(device=pl_module.device)
#
#         # Get model prediction ( need sigmoid since using BCEWithLogits )
#         logits = torch.sigmoid(pl_module(imgs1, imgs2))
#         threshold = torch.tensor([0.5])
#         preds = (logits > threshold).float() * 1
#
#         # Log the images as wandb Image
#         trainer.logger.experiment.log({
#             "examples": [wandb.Image(make_grid([img1, img2]), caption=f"Pred:{pred}, Label:{y}")
#                          for img1, img2, pred, y in zip(imgs1, imgs2, preds, labels)]
#         })
#         # log sampled images
#
#         self.logger.experiment.add_image('generated_images', grid, 0)
