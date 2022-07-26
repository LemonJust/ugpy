import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import pytorch_lightning as pl

# local imports
from ugpy.utils.loader import load_data, load_image
from ugpy.utils.data import Slices, Volumes, roi_to_centroids, split_to_rois, prepare_shift
from ugpy.utils.loader import load_centroids, load_labels, load_image, drop_unsegmented
from ugpy.utils.splitter import train_test_split
from ugpy.classification.augmentation import DataAugmentation

# TODO : rewrite stuff from os.join to Path
import os
import warnings


class CropDataModule(pl.LightningDataModule):
    """
    Prepares data for training, validation and testing.

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

        # whether to apply augmentation
        self.augment = config["augmentation"]
        self.transform = DataAugmentation(config)  # per batch augmentation_kornia

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.pin_memory = True
        # collects info for the logger
        self.info = {}

    def choose_dataset(self, image, centroids, labels):
        """
        Chooses the dataset based on the config file
        :return: custom Dataset
        """
        side = self.config["crop_side"]
        if self.config["input_type"] == "two slices":
            dataset = TwoSlicesDataset(image, side, centroids, labels=labels)
        elif self.config["input_type"] == "volume":
            shift = None
            if self.config["add_shifted"]:
                shift = prepare_shift(self.config["shift_by"], self.config["shift_labels"])
            dataset = VolumeDataset(image, side, centroids, labels=labels, shift=shift)

        else:
            raise ValueError(f"input_type can be 'two slices' or 'volume' only, but got {self.config['input_type']}")

        return dataset

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
        """

        self.data_dir = r"D:/code/repos/ugpy/data/test/gad1b"

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

                    print(f"Getting train data from {roi_id}")
                    train_datasets.append(self.choose_dataset(img, centroids_train, labels_train))
                    print(f"Getting validation data from {roi_id}")
                    val_datasets.append(self.choose_dataset(img, centroids_val, labels_val))

                    # keep track of total positive examples
                    total_pos_label_train = total_pos_label_train + np.sum(labels_train)
                    total_pos_label_val = total_pos_label_val + np.sum(labels_val)
            else:
                for roi_id in self.training_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    print(f"Getting train data from {roi_id}")
                    train_datasets.append(self.choose_dataset(img, centroids, labels))
                    # keep track of total positive examples
                    total_pos_label_train = total_pos_label_train + np.sum(labels)

                for roi_id in self.validation_roi_ids:
                    centroids, labels, img = load_data(self.data_dir, roi_id)
                    print(f"Getting validation data from {roi_id}")
                    val_datasets.append(self.choose_dataset(img, centroids, labels))
                    # keep track of total positive examples
                    total_pos_label_val = total_pos_label_val + np.sum(labels)

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
                print(f"Getting test data from {roi_id}")
                datasets.append(self.choose_dataset(img, centroids, labels))
                # keep track of total positive examples
                total_pos_label_test = total_pos_label_test + np.sum(labels)

            self.test_dataset = ConcatDataset(datasets)
            self.info.update({"num_train": len(self.test_dataset),
                              "frac1_test": total_pos_label_test / len(self.test_dataset)})

    def show_batch(self, win_size=(10, 10)):
        def _to_vis(imgs):
            img1 = imgs[:, :, 7, :, :]
            img2 = imgs[:, :, :, :, 7]
            imgs_sliced = torch.cat((img1, img2), dim=2)
            return imgs_sliced

        # get a batch from the training set: try with `val_datlaoader` :)
        imgs, labels = next(iter(self.train_dataloader()))
        imgs_aug = self.transform(imgs)  # apply transforms
        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs))
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(imgs_aug))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch

        if self.trainer.training and self.augment:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


class CropProbMapModule(pl.LightningDataModule):
    """
    Prepares data for making a probability map.
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

    def choose_dataset(self):
        """
        Chooses the dataset based on the config file
        :return: custom Dataset
        """
        side = self.config["crop_side"]
        if self.config["input_type"] == "two slices":
            dataset = TwoSlicesDataset.from_rois(self.image, side, self.rois)
        elif self.config["input_type"] == "volume":
            dataset = VolumeDataset.from_rois(self.image, side, self.rois)
        else:
            raise ValueError(f"input_type can be 'two slices' or 'volume' only, but got {self.config['input_type']}")

        return dataset

    def prepare_data(self):
        """
        prepares the data to be predicted
        """
        pass

    def setup(self, stage=None):
        if stage == "predict" or stage is None:
            print(f"Called setup 'stage == predict or stage is None' with {stage}")
            print(f"Getting predict data")
            self.pred_dataset = self.choose_dataset()
            print("Done")
            if self.batch_size == "all":
                self.batch_size = len(self.pred_dataset)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


# TODO : Create CropDataset and inherit TwoSlicesDataset and VolumeDataset from it
class TwoSlicesDataset(Dataset):
    """
    Crops slices through the center of the synapse
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


# TODO : Create CropDataset and inherit TwoSlicesDataset and VolumeDataset from it
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

    def __init__(self, img, side, centroids, labels=None, numba=True, shift=None):
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
        :param shift: dictionary with shift_list and shift_labels or None
        :type shift: dict
        :param numba: whether to use numba
        :type numba: bool
        """
        print(f"Creating dataset")
        self.numba = numba

        self.mean = np.mean(img)
        self.std = np.std(img)

        self.centroids = centroids

        if labels is not None:
            self.labels = torch.as_tensor(labels, dtype=self.label_dtype)
            self.synapse_centroids = centroids[labels]

            # record the fraction of synapses to not
            self.frac1 = np.sum(labels) / len(labels)
            # pick examples of synapses and not
            self.examples_idx = torch.as_tensor(self.get_examples(labels), dtype=torch.long)
        else:
            self.labels = None
            self.synapse_centroids = None

        if shift is not None:
            self._add_shifted_positive_examples(shift['shifts'], shift['labels'])

        self.side = side
        self.imgs = self.get_volumes(img, side)

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

    def _prepare_cropper(self, img, side):
        """
        Initialise cropper and get rid of any uncroppable centroids and labels.
        """

        # initialise slicer objects to make crops
        cropper = Volumes([side, side, side])

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

        return cropper

    def get_volumes(self, img, side):

        # initialise slicer objects to make crops
        cropper = self._prepare_cropper(img, side)

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

    def _add_shifted_positive_examples(self, shift_list, shift_labels):
        """
        Adds more synapse examples into the centroids
        by translating the centers according to each element in translation_list

        :param shift_list: how much to move centroids in zyx , Mx3 :[[dz0,dy0,dx0],[dz1,dy1,dx1], ...]
        dz dy dx can be positive or negative
        :type shift_list: list
        :param shift_labels: how to label shifted synapses - as synapses or not.
        You can provide label for each shift or if int, the same label is used for all shifts. ,
        for example: [1,0, ...] ( for first shift - synapse, for the second - not a synapse.
        :type shift_labels: Union(list, int)
        :return: None
        """

        # prepare labels
        if isinstance(shift_labels, int):
            shift_labels = [shift_labels] * len(shift_list)

        synapse_centroids = self.synapse_centroids
        n_syn = len(synapse_centroids)
        for label, shift in zip(shift_labels, shift_list):
            # move centroids according to the translation list,
            # add to the centroids and labels
            shifted = synapse_centroids + shift
            self.centroids = np.append(self.centroids, shifted, axis=0)
            self.labels = torch.cat([self.labels, torch.tensor([label] * n_syn, dtype=self.label_dtype)], dim=0)
            # add shifted centroids to synapse_centroids if the label is "1"
            if label:
                self.synapse_centroids = np.append(self.synapse_centroids, shifted, axis=0)

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


"""
_______________________________________________________________________________________________________________________
Tests 
_______________________________________________________________________________________________________________________
"""


# TODO : move tests to separate file


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
    data_dir = r"/data/test/gad1b"
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
    data_dir = r"/data/test/gad1b"
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


def check_examples2d():
    pl.seed_everything(222)
    data_dir = r"/data/test/gad1b"
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


def check_examples3d():
    pl.seed_everything(222)
    data_dir = r"/data/test/gad1b"
    side = 15

    roi_id = "1-1VWT"
    npz_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT.npz")
    csv_filename = os.path.join(data_dir, roi_id, "ROI_1-1VWT_synaptic_only.csv")
    img_filename = os.path.join(data_dir, roi_id, "Fish1_Nacre-like_47_LED_TP1-Done.ome.tiff")

    centroids = load_centroids(npz_filename)
    labels = load_labels(npz_filename, csv_filename)
    centroids, labels = drop_unsegmented(centroids, labels, x_max=500, z_max=90)
    img = load_image(img_filename)

    dataset = VolumeDataset(img, side, centroids, labels=labels)
    print(f"Examples: {dataset.examples_idx}")

    img, label = dataset[dataset.examples_idx]
    print(img.shape)
    print(label)

    # dataloader returns the nicer order of BxCxHxW
    dataloader = DataLoader(Subset(dataset, dataset.examples_idx), batch_size=6)
    img, label = next(iter(dataloader))
    print(img.shape)
    print(label)

    img1 = img[:, :, 7, :, :]
    img2 = img[:, :, :, :, 7]
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
    check_examples2d()
