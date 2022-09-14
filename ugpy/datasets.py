import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import pytorch_lightning as pl

# local imports
from loader import load_data, load_image
from preprocess import Slices, Volumes, roi_to_centroids, split_to_rois
from loader import load_centroids, load_labels, load_image, drop_unsegmented
from splitter import train_test_split

# TODO : rewrite stuff from os.join to Path
import os
import warnings

"""
Slices ________________________________________________________________________________________________________________
"""


class TwoSlicesDataset(Dataset):
    """
    Crops slices through the center of the
    """
    # what data type to use when converting images to tensors
    # TODO : compare float32 to float16
    image_dtype = torch.float32
    # what data type to use when converting labels to tensors
    """
    nn.CrossEntropyLoss expects its label input to be of type torch.Long and not torch.Float.
    Note that this behavior is opposite to nn.BCELoss where target is expected to be of the same type as the input.
    """
    # TODO : so what type should I use ?!?!
    label_dtype = torch.float32

    def __init__(self, img, side, centroids, labels=None, numba=True):
        """
        Creates a dataset that keeps two slices of size side x side for all the centroids,
         cropped  from the img at yx and zy orientation.

        :param img: image from which to crop slices
        :type img: numpy array
        :param side: the size of one side of the slice, slices are squares
        :type side: int
        :param centroids: coordinates of the centroids to crop in zyx order, Nx3 array
        :type centroids: numpy array
        :param labels: binary labels for each centroid
        :type labels: numpy array or list
        """
        print(f"Creating dataset")
        self.numba = numba

        self.mean = np.mean(img)
        self.std = np.std(img)

        self.centroids = centroids

        if labels is not None:
            self.examples_idx = torch.as_tensor(self.get_examples(labels), dtype=torch.long)
            self.frac1 = np.sum(labels) / len(labels)
            self.labels = torch.as_tensor(labels, dtype=self.label_dtype)
        else:
            self.labels = None

        self.imgs1, self.imgs2 = self.get_slices(img, side)

    @staticmethod
    def get_examples(labels):
        """
        Gets the indices of 3 images of class "1" and 3 images of class "0" to use as the examples
        """
        label_1 = np.random.choice(np.where(labels)[0], size=3,
                                   replace=False)
        label_0 = np.random.choice(np.where(np.logical_not(labels))[0], size=3,
                                   replace=False)
        return np.concatenate((label_1, label_0), axis=0)

    def get_slices(self, img, side):

        # initialise slicer objects to make crops
        slicer_yx = Slices([1, side, side])
        slicer_zy = Slices([side, side, 1])

        # check which centroids can be cropped with the currect box size
        cropable_yx = slicer_yx.get_cropable(self.centroids, img.shape)
        cropable_zy = slicer_zy.get_cropable(self.centroids, img.shape)
        cropable = np.logical_and(cropable_yx, cropable_zy)
        prc_cropable = np.sum(cropable) * 100 / len(cropable)
        print(f"Percent croppable: {prc_cropable}%")

        # drop uncropable centroids if needed
        if prc_cropable < 100:
            warnings.warn(f"Dropped {np.sum(~cropable)} uncropable centroids")
            self.centroids = self.centroids[cropable]
            if self.labels is not None:
                self.labels = self.labels[cropable]
                warnings.warn(f"Dropped {np.sum(~cropable)} uncropable labels")

        # crop
        if self.numba:
            # use acceleration by numba
            imgs1 = slicer_yx.crop_numba(self.centroids, img)
            imgs2 = slicer_zy.crop_numba(self.centroids, img)
        else:
            imgs1 = slicer_yx.crop(self.centroids, img)
            imgs2 = slicer_zy.crop(self.centroids, img)

        # apply normalisation
        imgs1 = self.standardize(imgs1)
        imgs2 = self.standardize(imgs2)

        # make tensors
        # torch.as_tensor doesn't use memory to create a copy
        imgs1 = torch.as_tensor(imgs1, dtype=self.image_dtype)
        imgs2 = torch.as_tensor(imgs2, dtype=self.image_dtype)

        return imgs1, imgs2

    def standardize(self, data):
        """
        Standardize data using z score per group. Volume or slices.
        Could do it with torch transforms... Is it better?
        """
        data = (data - self.mean) / self.std
        return data

    @classmethod
    def from_rois(cls, img, side, rois):
        """
        Creates a dataset from a roi : where each pixel in the roi is transformed into a centroid
        """
        print(f"Received rois: {rois}")
        centroids = roi_to_centroids(rois)
        print(f"getting {len(centroids)} centroids")
        return cls(img, side, centroids)

    def __getitem__(self, index):
        # This method should return only 1 sample and label
        # (according to "index"), not the whole dataset
        img1 = self.imgs1[index][None, :]
        img2 = self.imgs2[index][None, :]
        # TODO : if labels are not provided - returns only images
        #  shold I make mock labels instead?
        if self.labels is None:
            return (img1, img2)
        else:
            return (img1, img2), self.labels[index]

    def __len__(self):
        return len(self.centroids)

    def __repr__(self):
        info_str = f"""TwoSlices Dataset contains {len(self.centroids)} pairs of images"""
        return info_str

    def save(self):
        """
        Save the dataset to disk. You can save tensors and use them with TensorDataset as well...
        https://discuss.pytorch.org/t/save-dataset-into-pt-file/25293/5
        :return:
        :rtype:
        """
        pass


class TwoSlicesDataModule(pl.LightningDataModule):
    """
    For more info on LightningDataModule , see
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = True
        # collects info for the logger
        self.info = {}

    def prepare_data(self):
        """
        Downloading and saving data with multiple processes (distributed settings) will result in corrupted data.
        Lightning ensures the prepare_data() is called only within a single process,
        so you can safely add your downloading logic within. In case of multi-node training,
        the execution of this hook depends upon prepare_data_per_node.

        download,
        tokenize,
        etc…

        In my case I will create a full dataset here.

        Human segmentations:
        Zhuowei:
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VX8
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VXC
        Olivia:
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VWT
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VWW
        ML segmented, then human corrected:
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1WH8
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1WHA
        """

        self.data_dir = r"D:\Code\repos\UGPy\data\test\gad1b"
        self.side = self.config['slice_side']

        self.training_roi_ids = self.config['data_train']
        self.validation_roi_ids = self.config['data_valid']
        self.testing_roi_ids = self.config['data_test']
        # check if validation data is the same as train.
        # If true, will do train validation split. Else will load a separate dataset for validation.
        self.tv_split = set(self.training_roi_ids) == set(self.validation_roi_ids)

        # TODO : check here that these files exist in the data directory

    def setup(self, stage=None):
        """
        There are also data operations you might want to perform on every GPU. Use setup() to do things like:
        count number of classes,
        build vocabulary,
        perform train/val/test splits,
        create datasets,
        apply transforms (defined explicitly in your datamodule),
        etc…
        """
        if stage == "fit" or stage is None:
            print(f"Called setup 'stage == fit or stage is None' with {stage}")
            # example use:
            # self.train = Dataset(self.train_filenames, transform=True)
            # self.val = Dataset(self.val_filenames)

            train_datasets = []
            val_datasets = []
            # to calculate the portion of positive examples (for logging)
            total_pos_label_train = 0
            total_pos_label_val = 0

            # TODO : make a function to load centroids and append dataset
            if self.tv_split:
                print("Splitting each fish into train-validation datasets.")
                for roi_id in self.training_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    # do in a loop, to ensure that the same proportion of each fish is present in the validation
                    centroids_train, centroids_val, labels_train, labels_val = train_test_split(centroids, labels,
                                                                                                test_fraction=0.10)
                    total_pos_label_train = total_pos_label_train + np.sum(labels_train)
                    total_pos_label_val = total_pos_label_val + np.sum(labels_val)
                    print(f"Getting train data from {roi_id}")
                    train_datasets.append(TwoSlicesDataset(img, self.side, centroids_train, labels=labels_train))
                    print(f"Getting validation data from {roi_id}")
                    val_datasets.append(TwoSlicesDataset(img, self.side, centroids_val, labels=labels_val))
            else:
                for roi_id in self.training_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    total_pos_label_train = total_pos_label_train + np.sum(labels)
                    print(f"Getting train data from {roi_id}")
                    train_datasets.append(TwoSlicesDataset(img, self.side, centroids, labels=labels))
                for roi_id in self.validation_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    total_pos_label_val = total_pos_label_val + np.sum(labels)
                    print(f"Getting validation data from {roi_id}")
                    val_datasets.append(TwoSlicesDataset(img, self.side, centroids, labels=labels))

            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            self.info.update({"num_train": len(self.train_dataset),
                              "num_val": len(self.val_dataset),
                              "frac1_train": total_pos_label_train / len(self.train_dataset),
                              "frac1_val": total_pos_label_val / len(self.val_dataset)})

        if stage == "test" or stage is None:
            print(f"Called setup 'stage == test or stage is None' with {stage}")
            # example use:
            # self.test = Dataset(self.test_filenames)
            datasets = []
            total_pos_label_test = 0
            for roi_id in self.testing_roi_ids:
                centroids, labels, img = load_data(self.data_dir, roi_id)
                total_pos_label_test = total_pos_label_test + np.sum(labels)
                print(f"Getting test data from {roi_id}")
                datasets.append(TwoSlicesDataset(img, self.side, centroids, labels=labels))

            self.test_dataset = ConcatDataset(datasets)
            self.info.update({"num_train": len(self.test_dataset),
                              "frac1_test": total_pos_label_test / len(self.test_dataset)})

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


class TwoSlicesProbMapModule(pl.LightningDataModule):
    """
    Prepares data for prediction.
    """

    def __init__(self, rois, img, config):
        super().__init__()
        self.config = config

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = True

        self.rois = rois
        self.image = img
        # will be created by running prepare data
        self.pred_dataset = None

    def prepare_data(self):
        """
        prepares the data to be predicted
        """
        pass

    def setup(self, stage=None):
        if stage == "predict" or stage is None:
            print(f"Called setup 'stage == predict or stage is None' with {stage}")
            side = self.config["slice_side"]
            print(f"Getting predict data")
            self.pred_dataset = TwoSlicesDataset.from_rois(self.image, side, self.rois)
            print("Done")
            if self.batch_size == "all":
                self.batch_size = len(self.pred_dataset)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


"""
Volumes ______________________________________________________________________________________________________________
"""


class VolumeDataset(Dataset):
    """
    Crops volumes through the center of the
    """
    # what data type to use when converting images to tensors
    # TODO : compare float32 to float16
    image_dtype = torch.float32
    # what data type to use when converting labels to tensors
    """
    nn.CrossEntropyLoss expects its label input to be of type torch.Long and not torch.Float.
    Note that this behavior is opposite to nn.BCELoss where target is expected to be of the same type as the input.
    """
    # TODO : so what type should I use ?!?!
    label_dtype = torch.float32

    def __init__(self, img, side, centroids, labels=None, numba=True):
        """
        Creates a dataset that keeps volumes of size side x side x side for all the centroids,
         cropped  from the img at zyx orientation.

        :param img: image from which to crop slices
        :type img: numpy array
        :param side: the size of one side of the volume, volumes are cubes
        :type side: int
        :param centroids: coordinates of the centroids to crop in zyx order, Nx3 array
        :type centroids: numpy array
        :param labels: binary labels for each centroid
        :type labels: numpy array or list
        """
        print(f"Creating dataset")
        self.numba = numba

        self.mean = np.mean(img)
        self.std = np.std(img)

        self.centroids = centroids

        if labels is not None:
            self.examples_idx = torch.as_tensor(self.get_examples(labels), dtype=torch.long)
            self.frac1 = np.sum(labels) / len(labels)
            self.labels = torch.as_tensor(labels, dtype=self.label_dtype)
        else:
            self.labels = None

        self.imgs1, self.imgs2 = self.get_volumes(img, side)

    @staticmethod
    def get_examples(labels):
        """
        Gets the indices of 3 images of class "1" and 3 images of class "0" to use as the examples
        """
        label_1 = np.random.choice(np.where(labels)[0], size=3,
                                   replace=False)
        label_0 = np.random.choice(np.where(np.logical_not(labels))[0], size=3,
                                   replace=False)
        return np.concatenate((label_1, label_0), axis=0)

    def get_volumes(self, img, side):

        # initialise slicer objects to make crops
        cropper = Volumes([1, side, side])

        # check which centroids can be cropped with the currect box size
        cropable = cropper.get_cropable(self.centroids, img.shape)
        prc_cropable = np.sum(cropable) * 100 / len(cropable)
        print(f"Percent croppable: {prc_cropable}%")

        # drop uncropable centroids if needed
        if prc_cropable < 100:
            warnings.warn(f"Dropped {np.sum(~cropable)} uncropable centroids")
            self.centroids = self.centroids[cropable]
            if self.labels is not None:
                self.labels = self.labels[cropable]
                warnings.warn(f"Dropped {np.sum(~cropable)} uncropable labels")

        # crop
        if self.numba:
            # use acceleration by numba
            imgs = cropper.crop_numba(self.centroids, img)
        else:
            imgs = cropper.crop(self.centroids, img)

        # apply normalisation
        imgs = self.standardize(imgs)

        # make tensors
        # torch.as_tensor doesn't use memory to create a copy
        imgs = torch.as_tensor(imgs, dtype=self.image_dtype)

        return imgs

    def standardize(self, data):
        """
        Standardize data using z score per group. Volume or slices.
        Could do it with torch transforms... Is it better?
        """
        data = (data - self.mean) / self.std
        return data

    @classmethod
    def from_rois(cls, img, side, rois):
        """
        Creates a dataset from a roi : where each pixel in the roi is transformed into a centroid
        """
        print(f"Received rois: {rois}")
        centroids = roi_to_centroids(rois)
        print(f"getting {len(centroids)} centroids")
        return cls(img, side, centroids)

    def __getitem__(self, index):
        # This method should return only 1 sample and label
        # (according to "index"), not the whole dataset
        img = self.imgs[index][None, :]
        # TODO : if labels are not provided - returns only images
        #  shold I make mock labels instead?
        if self.labels is None:
            return img
        else:
            return img, self.labels[index]

    def __len__(self):
        return len(self.centroids)

    def __repr__(self):
        info_str = f"""TwoSlices Dataset contains {len(self.centroids)} pairs of images"""
        return info_str

    def save(self):
        """
        Save the dataset to disk. You can save tensors and use them with TensorDataset as well...
        https://discuss.pytorch.org/t/save-dataset-into-pt-file/25293/5
        :return:
        :rtype:
        """
        pass


class VolumeDataModule(pl.LightningDataModule):
    """
    For more info on LightningDataModule , see
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = True
        # collects info for the logger
        self.info = {}

    def prepare_data(self):
        """
        Downloading and saving data with multiple processes (distributed settings) will result in corrupted data.
        Lightning ensures the prepare_data() is called only within a single process,
        so you can safely add your downloading logic within. In case of multi-node training,
        the execution of this hook depends upon prepare_data_per_node.

        download,
        tokenize,
        etc…

        In my case I will create a full dataset here.

        Human segmentations:
        Zhuowei:
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VX8
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VXC
        Olivia:
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VWT
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1VWW
        ML segmented, then human corrected:
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1WH8
        https://synapse.isrd.isi.edu/chaise/record/#1/Zebrafish:Image%20Region/RID=1-1WHA
        """

        self.data_dir = r"D:\Code\repos\UGPy\data\test\gad1b"
        self.side = self.config['volume_side']

        self.training_roi_ids = self.config['data_train']
        self.validation_roi_ids = self.config['data_valid']
        self.testing_roi_ids = self.config['data_test']
        # check if validation data is the same as train.
        # If true, will do train validation split. Else will load a separate dataset for validation.
        self.tv_split = set(self.training_roi_ids) == set(self.validation_roi_ids)

        # TODO : check here that these files exist in the data directory

    def setup(self, stage=None):
        """
        There are also data operations you might want to perform on every GPU. Use setup() to do things like:
        count number of classes,
        build vocabulary,
        perform train/val/test splits,
        create datasets,
        apply transforms (defined explicitly in your datamodule),
        etc…
        """
        if stage == "fit" or stage is None:
            print(f"Called setup 'stage == fit or stage is None' with {stage}")
            # example use:
            # self.train = Dataset(self.train_filenames, transform=True)
            # self.val = Dataset(self.val_filenames)

            train_datasets = []
            val_datasets = []
            # to calculate the portion of positive examples (for logging)
            total_pos_label_train = 0
            total_pos_label_val = 0

            # TODO : make a function to load centroids and append dataset
            # if making train test split
            if self.tv_split:
                print("Splitting each fish into train-validation datasets.")
                for roi_id in self.training_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    # do in a loop, to ensure that the same proportion of each fish is present in the validation
                    centroids_train, centroids_val, labels_train, labels_val = train_test_split(centroids, labels,
                                                                                                test_fraction=0.10)
                    total_pos_label_train = total_pos_label_train + np.sum(labels_train)
                    total_pos_label_val = total_pos_label_val + np.sum(labels_val)
                    print(f"Getting train data from {roi_id}")
                    train_datasets.append(VolumeDataset(img, self.side, centroids_train, labels=labels_train))
                    print(f"Getting validation data from {roi_id}")
                    val_datasets.append(VolumeDataset(img, self.side, centroids_val, labels=labels_val))
            else:
                for roi_id in self.training_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    total_pos_label_train = total_pos_label_train + np.sum(labels)
                    print(f"Getting train data from {roi_id}")
                    train_datasets.append(VolumeDataset(img, self.side, centroids, labels=labels))
                for roi_id in self.validation_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    total_pos_label_val = total_pos_label_val + np.sum(labels)
                    print(f"Getting validation data from {roi_id}")
                    val_datasets.append(VolumeDataset(img, self.side, centroids, labels=labels))

            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
            self.info.update({"num_train": len(self.train_dataset),
                              "num_val": len(self.val_dataset),
                              "frac1_train": total_pos_label_train / len(self.train_dataset),
                              "frac1_val": total_pos_label_val / len(self.val_dataset)})

        if stage == "test" or stage is None:
            print(f"Called setup 'stage == test or stage is None' with {stage}")
            # example use:
            # self.test = Dataset(self.test_filenames)
            datasets = []
            total_pos_label_test = 0
            for roi_id in self.testing_roi_ids:
                centroids, labels, img = load_data(self.data_dir, roi_id)
                total_pos_label_test = total_pos_label_test + np.sum(labels)
                print(f"Getting test data from {roi_id}")
                datasets.append(VolumeDataset(img, self.side, centroids, labels=labels))

            self.test_dataset = ConcatDataset(datasets)
            self.info.update({"num_train": len(self.test_dataset),
                              "frac1_test": total_pos_label_test / len(self.test_dataset)})

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


class VolumeProbMapModule(pl.LightningDataModule):
    """
    Prepares data for prediction.
    """

    def __init__(self, rois, img, config):
        super().__init__()
        self.config = config

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = True

        self.rois = rois
        self.image = img
        # will be created by running prepare data
        self.pred_dataset = None

    def prepare_data(self):
        """
        prepares the data to be predicted
        """
        pass

    def setup(self, stage=None):
        if stage == "predict" or stage is None:
            print(f"Called setup 'stage == predict or stage is None' with {stage}")
            side = self.config["slice_side"]
            print(f"Getting predict data")
            self.pred_dataset = VolumeDataset.from_rois(self.image, side, self.rois)
            print("Done")
            if self.batch_size == "all":
                self.batch_size = len(self.pred_dataset)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


# TODO: CropDataModule --> to choose between volume and slice
"""
_______________________________________________________________________________________________________________________
Tests for the TwoSliceDataset and DataModule 
_______________________________________________________________________________________________________________________
"""


def check_probmap_module():
    img_file = "D:/Code/repos/UGPy/data/predict/prob_map/1-20FJ.tif"
    # use chunks to draw probability map in pieces
    zyx_chunks = [5, 5, 5]
    rois = split_to_rois(img_file, zyx_chunks)

    data_module = TwoSlicesProbMapModule(img_file, [rois[0]], 500)
    data_module.prepare_data()
    dataloader = data_module.predict_dataloader()
    (img1, img2) = next(iter(dataloader))
    print(img1.shape)
    print(img2.shape)


def check_combo_dataset():
    data_dir = r"D:\Code\repos\UGPy\data\test\gad1b"
    side = 15

    roi_id = "1-1VWT"
    npz_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT.npz")
    csv_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT_synaptic_only.csv")
    img_filename = os.path.join(data_dir, roi_id, "Fish1_Nacre-like_47_LED_TP1-Done.ome.tiff")

    centroids = load_centroids(npz_filename)
    labels = load_labels(npz_filename, csv_filename)
    centroids, labels = drop_unsegmented(centroids, labels, x_max=500, z_max=90)
    img = load_image(img_filename)

    dataset = TwoSlicesDataset(img, side, centroids, labels=labels)
    print(len(dataset))

    # ConcatDataset
    combo_dataset = ConcatDataset([dataset, dataset])
    print(len(combo_dataset))
    # access individual datasets like this:
    print(combo_dataset.datasets[0].mean)


def check_data_type():
    data_dir = r"D:\Code\repos\UGPy\data\test\gad1b"
    side = 15

    roi_id = "1-1VWT"
    npz_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT.npz")
    csv_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT_synaptic_only.csv")
    img_filename = os.path.join(data_dir, roi_id, "Fish1_Nacre-like_47_LED_TP1-Done.ome.tiff")

    centroids = load_centroids(npz_filename)
    labels = load_labels(npz_filename, csv_filename)
    centroids, labels = drop_unsegmented(centroids, labels, x_max=500, z_max=90)
    img = load_image(img_filename)

    dataset = TwoSlicesDataset(img, side, centroids, labels=labels)
    print(len(dataset))
    img1, img2, label = dataset[0]

    print(img2.shape)
    print(type(img2))
    print(img2.dtype)

    print(img1.shape)
    print(type(img1))
    print(img1.dtype)

    print(type(label))
    print(label.dtype)


def check_examples():
    data_dir = r"D:\Code\repos\UGPy\data\test\gad1b"
    side = 15

    roi_id = "1-1VWT"
    npz_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT.npz")
    csv_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT_synaptic_only.csv")
    img_filename = os.path.join(data_dir, roi_id, "Fish1_Nacre-like_47_LED_TP1-Done.ome.tiff")

    centroids = load_centroids(npz_filename)
    labels = load_labels(npz_filename, csv_filename)
    centroids, labels = drop_unsegmented(centroids, labels, x_max=500, z_max=90)
    img = load_image(img_filename)

    dataset = TwoSlicesDataset(img, side, centroids, labels=labels)
    print(dataset.examples_idx)

    (img1, img2), label = dataset[dataset.examples_idx]
    print(img1.shape)
    print(label)

    # dataloader returns the nicer order of BxCxHxW
    dataloader = DataLoader(Subset(dataset, dataset.examples_idx), batch_size=6)
    (img1, img2), label = next(iter(dataloader))
    print(img1.shape)
    print(label)

    imgs = torch.cat((img1, img2), dim=2)
    print(imgs.shape)

    fig = plt.figure(figsize=(6., 3.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     share_all=True)

    for i_cell, ax in enumerate(grid):
        # Iterating over the grid returns the Axes.
        ax.imshow(torch.squeeze(imgs[i_cell]))
        ax.set_title(f"Label\n{label[i_cell]}")
    plt.show()


if __name__ == "__main__":
    check_probmap_module()
