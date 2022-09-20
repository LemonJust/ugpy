"""
Based on stuff from here :
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/augmentation_kornia.html
"""
import torch
from torch import Tensor
import torch.nn as nn


# import torchio as tio


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, config) -> None:
        super().__init__()

        self.transforms = self.choose_transform(config)

    @staticmethod
    def choose_transform(config):
        if config["input_type"] == "two slices":
            transforms = None
        elif config["input_type"] == "volume":
            # there is a 25% chance that none of them will be applied
            transforms = RandomVolumeFlip(p=0.7)
        else:
            raise ValueError(f"input_type can be 'two slices' or 'volume' only, but got {config['input_type']}")

        return transforms

    @torch.no_grad()  # disable gradients for efficiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = torch.stack([self.transforms(instance) for instance in x])  # BxCxHxW
        return x_out


# class Preprocess(nn.Module):
#     """Module to perform pre-process using Kornia on torch tensors."""
#
#     @torch.no_grad()  # disable gradients for effiency
#     def forward(self, x) -> Tensor:
#         x_tmp: np.ndarray = np.array(x)  # HxWxC
#         x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
#         return x_out.float() / 255.0

class RandomVolumeFlip(nn.Module):
    """Horizontally flip the given volume randomly with a given probability.
    Input Tensor, is expected to have [C, D, H, W] shape

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.n_flips = 7

    def forward(self, vol):
        """
        Args:
            vol (Tensor): Volume to be flipped.

        Returns:
            Tensor: Randomly flipped volume.
        """
        to_flip = torch.rand(1)  # or not to flip... that is the question ... also how to flip

        flip_type_split = torch.linspace(0, self.p, self.n_flips + 1)
        if to_flip < self.p:
            # flip around individual axis
            if flip_type_split[0] <= to_flip < flip_type_split[1]:
                return torch.flip(vol, [1])
            if flip_type_split[1] <= to_flip < flip_type_split[2]:
                return torch.flip(vol, [2])
            if flip_type_split[2] <= to_flip < flip_type_split[3]:
                return torch.flip(vol, [3])
            # flip around 2 axis
            if flip_type_split[3] <= to_flip < flip_type_split[4]:
                return torch.flip(vol, [1, 2])
            if flip_type_split[4] <= to_flip < flip_type_split[5]:
                return torch.flip(vol, [1, 3])
            if flip_type_split[5] <= to_flip < flip_type_split[6]:
                return torch.flip(vol, [2, 3])
            # flip around all axis
            if flip_type_split[6] <= to_flip < flip_type_split[7]:
                return torch.flip(vol, [1, 2, 3])
        return vol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
