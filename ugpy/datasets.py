import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import pytorch_lightning as pl

# local imports
from loader import load_data
from preprocess import Slices, roi_to_centroids
from loader import load_centroids, load_labels, load_image, drop_unsegmented
from splitter import train_test_split

# TODO : rewrite stuff from os.join to Path
import os


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

    def __init__(self, img, side, centroids, labels=None):
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

        self.mean = np.mean(img)
        self.std = np.std(img)
        self.examples_idx = torch.as_tensor(self.get_examples(labels), dtype=torch.long)

        self.centroids = centroids

        self.imgs1, self.imgs2 = self.get_slices(img, side)

        if labels is not None:
            self.frac1 = np.sum(labels)/len(labels)
            self.labels = torch.as_tensor(labels, dtype=self.label_dtype)
        else:
            self.labels = None

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
        slicer_yx = Slices([1, side, side])
        slicer_zy = Slices([side, side, 1])

        imgs1 = slicer_yx.crop(self.centroids, img)
        imgs2 = slicer_zy.crop(self.centroids, img)

        imgs1 = self.standardize(imgs1)
        imgs2 = self.standardize(imgs2)

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
        centroids = roi_to_centroids(rois)
        return cls(img, side, centroids)

    def __getitem__(self, index):
        # This method should return only 1 sample and label
        # (according to "index"), not the whole dataset
        img1 = self.imgs1[index][None, :]
        img2 = self.imgs2[index][None, :]

        if self.labels is None:
            return img1, img2
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

    def __init__(self, batch_size, num_workers=1, pin_memory=True):
        super().__init__()

        self.batch_size = batch_size

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        self.num_classes = 2
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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

        data_dir = r"D:\Code\repos\UGPy\data\test\gad1b"
        side = 15

        self.training_roi_ids = ["1-1WHA", "1-1VWT"]
        self.testing_roi_ids = ["1-1VXC"]

        train_datasets = []
        val_datasets = []
        # to calculate the portion of positive examples (for logging)
        num1_train = 0
        num1_val = 0
        for roi_id in self.training_roi_ids:
            print(roi_id)
            centroids, labels, img = load_data(data_dir, roi_id)
            # to ensure that the same proportion of each fish is present in the validation
            # TODO : do I need to worry about the tp1 and tp2 of the same fish
            #  being present in the val and training data?
            #  should I only use tp1 ( or 2 ) for a given fish in validation ?
            centroids_train, centroids_val, labels_train, labels_val = train_test_split(centroids, labels,
                                                                                        test_fraction=0.10)
            num1_train = num1_train + np.sum(labels_train)
            num1_val = num1_val + np.sum(labels_val)
            train_datasets.append(TwoSlicesDataset(img, side, centroids_train, labels=labels_train))
            val_datasets.append(TwoSlicesDataset(img, side, centroids_val, labels=labels_val))
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)
        self.frac1_train = num1_train/len(self.train_dataset)
        self.frac1_val = num1_val / len(self.val_dataset)

        datasets = []
        num1_test = 0
        for roi_id in self.testing_roi_ids:
            print(roi_id)
            centroids, labels, img = load_data(data_dir, roi_id)
            num1_test = num1_test + np.sum(labels)
            datasets.append(TwoSlicesDataset(img, side, centroids, labels=labels))
        self.test_dataset = ConcatDataset(datasets)
        self.frac1_test = num1_test / len(self.test_dataset)


    def setup(self, stage=None):
        """
        There are also data operations you might want to perform on every GPU. Use setup() to do things like:
        count number of classes,
        build vocabulary,
        perform train/val/test splits,
        create datasets,
        apply transforms (defined explicitly in your datamodule),
        etc…

        Nothing to do here for me because I already did everything in the prepare_data
        ( due to how my data is made of individual fish, I had to do it that way)
        """
        # TODO : maybe split based on fish + label here and get a full_dataset in the prepare data ?
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


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
    check_examples()
