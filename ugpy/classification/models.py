# PyTorch libraries
import torch
from torch import nn
from torchinfo import summary


def conv2d_module(in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True):
    """
    define a CONV2D => BN2D => RELU pattern
    batch_norm : weather or not to apply batch norm
    """
    if batch_norm:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU())
    # return the block
    return conv


def conv3d_module(in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True):
    """
    define a CONV3D => BN2D => RELU pattern
    batch_norm : weather or not to apply batch norm
    """
    if batch_norm:
        conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
    else:
        conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU())
    # return the block
    return conv


# TODO : add zero padding ????
class TwoBranchConv2d(nn.Module):
    """
    Model with two branches of 2d convolution along them
    https://stackoverflow.com/questions/66786787/pytorch-multiple-branches-of-a-model
    """

    def __init__(self, do_batch_norm=False):
        super(TwoBranchConv2d, self).__init__()

        self.cnns = nn.ModuleList([self.cnn_branch(do_batch_norm=do_batch_norm),
                                   self.cnn_branch(do_batch_norm=do_batch_norm)])
        # NOTE : it's important to use nn.BCEWithLogitsLoss loss when having no sigmoid ...
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # skipping sigmoid layer because during the training wea are using nn.BCEWithLogitsLoss ,
            # which combines nn.Sigmoid and nn.BCELoss
        )

    @staticmethod
    def cnn_branch(do_batch_norm=False):
        conv = nn.Sequential(
            conv2d_module(1, 4, 3, stride=1, padding=0, batch_norm=do_batch_norm),
            conv2d_module(4, 4, 3, stride=1, padding=0, batch_norm=do_batch_norm),
            nn.MaxPool2d(2),
            conv2d_module(4, 8, 3, stride=1, padding=0, batch_norm=do_batch_norm),
            conv2d_module(8, 8, 3, stride=1, padding=0, batch_norm=do_batch_norm)
        )
        return conv

    def forward(self, x):
        """
        :param x: a tuple of images in xy and zy orientation
        :type x: tuple(tensor, tensor)
        :return: value at the final node, without sigmoid applied
        :rtype: tensor
        """
        x0 = self.cnns[0](x[0])
        x1 = self.cnns[1](x[1])

        combined = torch.cat((x0.view(x0.size(0), -1),
                              x1.view(x1.size(0), -1)), dim=1)

        # before squeeze it is (Batch x 1) , but need to match labels (Batch)
        z = torch.squeeze(self.fc(combined))
        return z


class Conv3d(nn.Module):
    """
    Model with 3D input
    Expects input 15x15x15 pixels
    https://stackoverflow.com/questions/66786787/pytorch-multiple-branches-of-a-model
    """

    def __init__(self, do_batch_norm=False):
        super(Conv3d, self).__init__()

        self.cnns = self.cnn_branch(do_batch_norm=do_batch_norm)
        # NOTE : it's important to use nn.BCEWithLogitsLoss loss when having no sigmoid ...
        self.fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # skipping sigmoid layer because during the training wea are using nn.BCEWithLogitsLoss ,
            # which combines nn.Sigmoid and nn.BCELoss
        )

    @staticmethod
    def cnn_branch(do_batch_norm=False):
        conv = nn.Sequential(
            conv3d_module(1, 4, 3, stride=1, padding=0, batch_norm=do_batch_norm),
            conv3d_module(4, 4, 3, stride=1, padding=0, batch_norm=do_batch_norm),
            nn.MaxPool3d(2),
            conv3d_module(4, 8, 3, stride=1, padding=0, batch_norm=do_batch_norm),
            conv3d_module(8, 8, 3, stride=1, padding=0, batch_norm=do_batch_norm)
        )
        return conv

    def forward(self, x):
        """
        :param x: a tuple of images in xy and zy orientation
        :type x: tuple(tensor, tensor)
        :return: value at the final node, without sigmoid applied
        :rtype: tensor
        """
        x = self.cnns(x)
        flattened = torch.flatten(x, start_dim=1)
        # # before squeeze it is (Batch x 1) , but need to match labels (Batch)
        x = torch.squeeze(self.fc(flattened))
        return x


class TinyConv3d(Conv3d):
    """
    Model with 3D input
    Expects input 7x7x7 pixels
    """

    def __init__(self, do_batch_norm=False):
        super(TinyConv3d, self).__init__()

        self.cnns = self.cnn_branch(do_batch_norm=do_batch_norm)

    @staticmethod
    def cnn_branch(do_batch_norm=False):
        """
        Padding is actually important:
        https://stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad
        https://cs231n.github.io/convolutional-networks/
        """
        conv = nn.Sequential(
            conv3d_module(1, 4, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            conv3d_module(4, 4, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            nn.MaxPool3d(2),
            conv3d_module(4, 8, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            conv3d_module(8, 8, 3, stride=1, padding=0, batch_norm=do_batch_norm)
        )
        return conv


class SuperTinyConv3d(Conv3d):
    """
    Model with 3D input
    Expects input 5x5x5 pixels
    """

    def __init__(self, do_batch_norm=False):
        super(SuperTinyConv3d, self).__init__()

        self.cnns = self.cnn_branch(do_batch_norm=do_batch_norm)

    @staticmethod
    def cnn_branch(do_batch_norm=False):
        """
        Padding is actually important:
        https://stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad
        https://cs231n.github.io/convolutional-networks/
        """
        conv = nn.Sequential(
            conv3d_module(1, 4, 3, stride=1, padding=2, batch_norm=do_batch_norm),
            conv3d_module(4, 4, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            nn.MaxPool3d(2),
            conv3d_module(4, 8, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            conv3d_module(8, 8, 3, stride=1, padding=0, batch_norm=do_batch_norm)
        )
        return conv


class SuperTinyFC(nn.Module):
    """
    Model with all fully connected layers
    expects 1x5x5x5 input
    """

    def __init__(self, do_batch_norm=False):
        super(SuperTinyFC, self).__init__()

        # NOTE : it's important to use nn.BCEWithLogitsLoss loss when having no sigmoid ...
        self.fc = nn.Sequential(
            nn.Linear(125, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # skipping sigmoid layer because during the training wea are using nn.BCEWithLogitsLoss ,
            # which combines nn.Sigmoid and nn.BCELoss
        )

    def forward(self, x):
        """
        :param x: a tuple of images in xy and zy orientation
        :type x: tuple(tensor, tensor)
        :return: value at the final node, without sigmoid applied
        :rtype: tensor
        """
        flattened = torch.flatten(x, start_dim=1)
        # before squeeze it is (Batch x 1) , but need to match labels (Batch)
        z = torch.squeeze(self.fc(flattened))
        return z


class TinyConv3d_Localization(Conv3d):
    """
    Model with 3D input
    Expects input 7x7x7 pixels
    """

    def __init__(self, do_batch_norm=False):
        super(TinyConv3d_Localization, self).__init__()

        self.cnns = self.cnn_branch(do_batch_norm=do_batch_norm)

    @staticmethod
    def cnn_branch(do_batch_norm=False):
        """
        Padding is actually important:
        https://stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad
        https://cs231n.github.io/convolutional-networks/
        """
        conv = nn.Sequential(
            conv3d_module(1, 4, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            conv3d_module(4, 4, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            nn.MaxPool3d(2),
            conv3d_module(4, 8, 3, stride=1, padding=1, batch_norm=do_batch_norm),
            conv3d_module(8, 8, 3, stride=1, padding=0, batch_norm=do_batch_norm)
        )
        return conv



if __name__ == "__main__":
    # CDHW
    model = SuperTinyFC()
    batch_size = 16
    summary(model, input_size=(batch_size, 1, 5, 5, 5))
