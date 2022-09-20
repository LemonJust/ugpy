"""
Core classes to del with my type of data
"""

import numpy as np
from tifffile import TiffFile
import tifffile as tif
from numba import njit
import scipy.ndimage as nd


class Cropper:
    """
    Takes care of cropping 3D volumes or 2D slices from an image, going through the center.
    """

    def __init__(self, shape):
        """
        shape : [z, y, x] size of the slice, in pixels
        centroids : [z, y, x] center of the slice, in pixels
        img: 3D numpy array for an image
        """
        # shape as a 3d and a 2d array (one dimension is 1)
        assert np.sum(shape == 1) <= 1, "Shape must be 3D or 2D : no dimensions of size 1," \
                                        " or exactly one such dimension"
        assert np.sum(shape == 0) < 1, "Shape must be 3D or 2D : no 0 dimensions"

        self.shape_3d = np.array(shape)

    def crop(self, centroids, img, return_status=False):
        """
        img: 3D array , image from which to crop
        centroids : N x 3 array
        Returns
        imgs: array N_dim1_dim2 with image slices through the centroids or all zeros if crop was not successful
        crop_status: weather or not the centroid was cropped
        """
        n_centroids = centroids.shape[0]
        imgs = np.zeros((n_centroids, self.shape_3d[0], self.shape_3d[1], self.shape_3d[2]))
        crop_status = np.ones((n_centroids, 1)).astype('bool')

        for i, centroid in enumerate(centroids):
            start = centroid - self.shape_3d // 2
            end = centroid + self.shape_3d // 2 + 1
            # check that crop is fully inside the image:
            if np.any(start < 0) or np.any(end > img.shape):
                crop_status[i] = False
            else:
                imgs[i, :] = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        if return_status:
            # return images with dimension of size 1 (if 2d case) removed and a list of cropped and not
            return np.squeeze(imgs), crop_status
        else:
            return np.squeeze(imgs)

    def crop_numba(self, centroids, img):
        """
        crops without checking if the crop is possible.
        But much-much aster than crop, but you need to be careful
        """

        @njit
        def crop_numba_loop(centroids, img, imgs, half_shape_3d):
            for i, centroid in enumerate(centroids):
                start = centroid - half_shape_3d
                end = centroid + half_shape_3d + 1
                # check that crop is fully inside the image:
                crop = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                imgs[i, :] = crop
            return imgs

        n_centroids = centroids.shape[0]
        imgs = np.zeros((n_centroids, self.shape_3d[0], self.shape_3d[1], self.shape_3d[2]))
        half_shape_3d = self.shape_3d // 2
        imgs = crop_numba_loop(centroids, img, imgs, half_shape_3d)
        return np.squeeze(imgs)

    def get_cropable(self, centroids, img_shape, as_idx=False):
        """
        Returns a boolean list, with false if the centroid is too close to the boarder of the image
        and can't be cropped.
        as_idx : if True will return the indexes of cropable centroids , if False will return a boolean array
        """

        start = centroids - self.shape_3d // 2
        end = centroids + self.shape_3d // 2 + 1
        # check that volumes are fully inside the image:
        cropable = ~np.logical_or(np.any(start < 0, axis=1),
                                  np.any(end > img_shape, axis=1))
        if as_idx:
            return np.where(cropable)[0]
        else:
            return cropable


class Slices(Cropper):
    """
    Takes care of 2D slices from an image, going through the center.
    """

    def __init__(self, shape):
        """
        shape : [z, y, x] size of the slice, in pixels
        centroids : [z, y, x] center of the slice, in pixels
        img: 3D numpy array for an image
        """
        # TODO : make it possible to initialise with a 2d shape + orientation

        super().__init__(shape)
        self.shape_2d = self.get_shape_2d()
        self.orientation = self.get_orientation()

    def get_shape_2d(self):
        """
        Returns 2d shape by dropping the 0 dimension.
        """
        if np.sum(self.shape_3d == 1) == 1:
            return self.shape_3d[self.shape_3d != 1]
        else:
            return None

    def get_orientation(self):
        """
        Names the orientation based on shape.
        """
        # figure out slice orientation by looking at  what dimension is missing
        orient_list = ['yx', 'zx', 'zy']
        is_0 = np.where(self.shape_3d == 1)[0][0]
        return orient_list[is_0]

    def flip(self, dim):
        """
        flips slices for data augmentation.
        dim: dimension to flip , 0 or 1
        """
        pass

    def rotate_90(self):
        """
        Rotates slices by 90 deg for data augmentation. Only for 'xy' slices
        """
        pass


class Volumes(Cropper):
    """
    Takes care of 3D blocks from an image, going through the center.
    """

    def __init__(self, shape):
        """
        shape : [z, y, x] size of the volume, in pixels
        centroids : [z, y, x] centers of the volumes, in pixels
        img: 3D numpy array for an image
        """
        super().__init__(shape)
        self.orientation = '3d'

    def flip(self, dim):
        """
        flips slices for data augmentation.
        dim: dimension to flip , 0, 1 or 2
        """
        pass

    def rotate_90(self):
        """
        Rotates volumes by 90 deg around z axis for data augmentation ( rotation in XY plane)
        """
        pass


class Image:
    def __init__(self, resolution, filename=None, img=None, info=None, mask=None):
        """
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional
        ( fields can be empty if don't need to crop the corresponding dimention).
        """

        assert filename is not None or img is not None, "Provide filename or img."
        assert (filename is None) is not (img is None), "Provide exactly one of the following: filename or img."

        self.filename = filename
        self.resolution = np.array(resolution)
        self.info = info

        if img is None:
            self.img = self.read_image()
        else:
            self.img = img

        self.shape = self.img.shape

        self.mask = mask
        if self.mask is not None:
            self.mask = self.crop(mask)

        self.shape = self.img.shape

    def read_image(self):
        """
        Reads image in ZYX order.
        """""
        img = tif.imread(self.filename)
        return img

    def crop(self, mask):
        """
        Crops an image: drops everything outside a rectangle mask (in pixels) and remembers the parameters of the crop.
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional ( fields can be empty if don't need to crop there).
        """

        for key in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
            if key not in mask:
                mask[key] = None

        self.img = self.img[
                   mask['zmin']:mask['zmax'],
                   mask['ymin']:mask['ymax'],
                   mask['xmin']:mask['xmax']
                   ]

        self.shape = self.img.shape

        return mask

    def blur(self, sigma):
        """
        sigma : gaussian filter parameters, ZYX order , in pixels
        (from scipy : scalar or sequence of scalars Standard deviation for Gaussian kernel. The
        standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.)
        """
        img = nd.gaussian_filter(self.img, sigma)
        return Image(self.resolution, img=img, info=f"Blurred {self.filename}")

    def dog(self, sigma1, sigma2):
        """
        sigma1, sigma2 : gaussian filter parameters, ZYX order , in pixels
        (from scipy : scalar or sequence of scalars Standard deviation for Gaussian kernel. The
        standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.)
        """
        img = nd.gaussian_filter(self.img, sigma2) - nd.gaussian_filter(self.img, sigma1)
        return Image(self.resolution, img=img, info=f"DoG {self.filename}")

    def threshold(self, thr, binary=True):
        """
        Image threshold. Returns an image.
        binary : wether to return a binary mask or the original pixel values ( only the ones above the threshold ).
        """
        img = (self.img > thr)

        if not binary:
            img = self.img * img

        return Image(self.resolution, img=img.astype(np.int16), info=f"Threshold {self.filename}, thr = {thr}")

    # def local_max(self, order):
    #     """
    #     Finds local maxima in Image.
    #     Returns points.
    #     """
    #     # local_maxima_3D from utils
    #     coords, values = local_maxima_3D(self.img, order=order)
    #     return Points(coords, units='pix', resolution=self.resolution, info={'values': values})

    def imwrite(self, filename):
        """
        Saves image to disc as tif.
        """
        # ImageJ hyperstack axes must be in TZCYXS order...
        # it ignores my 'axis' metadata (Is it called something else?).. so just expand to ZCYX
        tif.imwrite(filename, np.expand_dims(self.img, axis=1).astype(np.uint16), imagej=True)

    @staticmethod
    def get_image_shape(img_file):
        # get file info
        stack = TiffFile(img_file, _multifile=False)
        z_size = len(stack.pages)
        page = stack.pages.get(0)
        y_size, x_size = page.shape
        stack.close()
        return (z_size, y_size, x_size)


class Points:
    """ Points class represents and manipulates xyz coords. """

    def __init__(self, zyx_arr, units='pix', resolution=None, idx=None, info=None):
        """ Create a new point at the origin
        units : in what units the zyx_arr coordinates are given. Can be 'pix' or 'phs'
        for pixels or physical units respectively.
        info : dictionary with lists or numpy arrays, specifying property per point.
        """

        if resolution is None:
            resolution = [1, 1, 1]
        self.resolution = np.array(resolution)

        self.zyx = {}
        if units == 'pix':
            self.zyx['pix'] = np.array(zyx_arr)
            self.zyx['phs'] = self.zyx['pix'] * self.resolution
        elif units == 'phs':
            self.zyx['phs'] = np.array(zyx_arr)
            self.zyx['pix'] = np.round(self.zyx['phs'] / self.resolution)

            # personal id for each point
        self.num_points = self.zyx['pix'].shape[0]
        if idx is None:
            self.idx = np.arange(self.num_points)
        else:
            self.idx = np.array(idx)

        self.info = info

    def __repr__(self):
        return f'Number of points : {self.num_points}\nResolution : {self.resolution}\nCoordinates' \
               f' :\n- pixels\n{self.zyx["pix"]}\n- physical units\n{self.zyx["phs"]}'

    @classmethod
    def from_dict(cls, d):
        """
        Load Points object from a dictionary.
        TODO : maybe you want to save and load both pix and phs units ... using phs only for now
        """
        # info might be missing
        if 'info' in d:
            info = d['info']
        else:
            info = None

        points = cls(d['zyx'], units='phs', resolution=d['resolution'], idx=d['idx'], info=info)
        return points

    @classmethod
    def from_json(cls, filename):
        """
        Load Points object from json file.
        TODO : maybe you want to save and load both pix and phs units ... using phs only for now
        """
        # create an object for the class to return
        with open(filename) as json_file:
            j = json.load(json_file)
        points = cls.from_dict(j)
        return points

    @classmethod
    def from_predictions(cls, filename, prob_thr=0.5, resolution=[1, 1, 1], units='pix'):
        df = pd.read_csv(filename)
        points = cls(df[['Z', 'Y', 'X']][df["prob"] > prob_thr].to_numpy(),
                     units=units, resolution=resolution)
        return points

    def to_dict(self):
        """
        Transform Points object into json format and save as a file.
        """

        d = {"resolution": self.resolution.tolist(),
             "zyx": self.zyx['phs'].tolist(),
             "idx": self.idx.tolist()}

        if self.info is not None:
            d["info"] = {key: self.info[key].tolist() for key in self.info}

        return d

    def to_json(self, filename):
        """
        Transform Points object into json format and save as a file.
        """
        j = json.dumps(self.to_dict())

        with open(filename, 'w') as json_file:
            json_file.write(j)

    def crop(self, mask, units='pix'):
        """
        Crops a point cloud: drops everything outside a rectangle mask (in pixels or physical units)
        and remembers the parameters of the crop.
        mask : dict with xmin, xmax, ymin, ymax, zmin, zmax optional ( fields can be empty if don't need to crop there).

        """
        # calculate the crop
        is_in = np.ones(self.num_points, dtype=bool)

        for ikey, key in enumerate(['zmin', 'ymin', 'xmin']):
            if key in mask and mask[key] is not None:
                is_in = np.logical_and(is_in,
                                       mask[key] < self.zyx[units][:, ikey])
        for ikey, key in enumerate(['zmax', 'ymax', 'xmax']):
            if key in mask and mask[key] is not None:
                is_in = np.logical_and(is_in,
                                       self.zyx[units][:, ikey] < mask[key])
        # apply crop
        zyx = self.zyx[units][is_in, :]
        idx = self.idx[is_in]
        if self.info is not None:
            info = {key: np.array(self.info[key])[is_in] for key in self.info}

        points = Points(zyx, units=units, resolution=self.resolution, idx=idx, info=info)
        return points

    def recenter(self, center, units='pix'):
        """
        Sets the zero to center ( array of 3 elements in zyx order ).
        Center needs to be in pixels or the same physical units as the pointcloud.
        """
        center = np.array(center)
        zyx = self.zyx[units] - center

        points = Points(zyx, units=units, resolution=self.resolution, idx=self.idx)
        return points

    def transform(self, transform, units='phs'):
        """
        Applies transform to points in given units , default to physical.
        transform : AffineTransform, a matrix and a center representing an affine transform in 3D.
        In such format, that to apply transform matrix to a set of zyx1 points : zyx1@transform.matrix .

        Returns Points with the same type of dta as the original, but coordinates transformed.
        """

        def to_zyx1(zyx_arr):
            n_points = zyx_arr.shape[0]
            ones = np.ones(n_points)
            return np.c_[zyx_arr, ones[:, np.newaxis]]

        zyx = self.zyx[units] - transform.center
        zyx1 = to_zyx1(zyx)
        transformed_zyx1 = zyx1 @ transform.matrix
        transformed_zyx = transformed_zyx1[:, 0:3] + transform.center

        points = Points(transformed_zyx, units=units, resolution=self.resolution, idx=self.idx, info=self.info)
        return points

    def fit_block(self, blc, padding=[0, 0, 0]):
        """
        Takes a ptc and crops it to block.
        padding : in pixels (in the pixel space of the block)
        """
        # get mask in physical units :
        start = (blc.start - padding) * blc.img.resolution
        end = (blc.start + blc.size + padding) * blc.img.resolution
        mask = {'zmin': start[0], 'zmax': end[0],
                'ymin': start[1], 'ymax': end[1],
                'xmin': start[2], 'xmax': end[2]}

        return self.crop(mask, units='phs')

    def fit_transform(self, af, padding=[0, 0, 0]):
        """
        Takes a ptc and crops it to the area on which affine transform was calculated.
        padding : in physical units? in zyx order
        Assumes the transform is in physical units.
        """
        # TODO : maybe make transform carry the UNITS
        # TODO : make transform carry FIXED info ... now works ONLY because fixed and moving are the same

        # get mask in physical units :
        start = af.center - padding
        end = af.center + af.size + padding
        mask = {'zmin': start[0], 'zmax': end[0],
                'ymin': start[1], 'ymax': end[1],
                'xmin': start[2], 'xmax': end[2]}

        return self.crop(mask, units='phs')

    def split(self, blocks, padding=[0, 0, 0]):
        """ Splits points into Blocks
        Creates a points list in the order, that corresponds to the given blocks list.
        """
        points = []
        for block in blocks:
            points.append(self.fit_block(block, padding))
        return points

    @classmethod
    def concat(cls, ptc_list):
        """
        combines point clouds in ptc_list into one, concatenating the coordinates and idx.
        all point clouds need to have the same resolution.
        padding : zyx padding in pixels or phs
        """

        resolution = ptc_list[0].resolution
        zyx = None
        idx = None

        for i_ptc, ptc in enumerate(ptc_list):
            if i_ptc == 0:
                zyx = ptc.zyx['phs']
                idx = ptc.idx
            else:
                assert np.all(resolution == ptc.resolution), "Resolution should be the same for all point clouds"
                zyx = np.r_[zyx, ptc.zyx['phs']]
                idx = np.r_[idx, ptc.idx]

        # TODO : add info as well
        points = cls(zyx, units='phs', resolution=resolution, idx=idx)
        return points

    def pw_transform(self, transfom_list):
        # TODO : remake AffineTransforms to have info about the fixed as well as the moving
        """
        Piece-wise transforms the ptc according to each block alignment.
        Creates a points list in the order, that corresponds to the given transfom list.
        Each ptc in the list contains all the points, but transformed according to the different alignments.
        """
        points = []
        for af in transfom_list:
            # transform already takes the top left corner into account (center)
            ptc = self.transform(af, units='phs')
            points.append(ptc)

        return points

    def filter_by_info(self, feature, filter, units='phs'):
        """
        Applies threshold on the specified info.
        filter: 'max' and 'min' values to keep.
        feature: what info to use
        """
        assert self.info is not None, "Can't filter by info : info is None"

        is_in = np.ones((self.num_points,)).astype(np.bool)
        if 'max' in filter:
            is_in = np.logical_and(is_in, (np.array(self.info[feature]) <= filter['max']))
        if 'min' in filter:
            is_in = np.logical_and(is_in, (np.array(self.info[feature]) >= filter['min']))

        zyx = self.zyx[units][is_in, :]
        idx = self.idx[is_in]
        info = {key: np.array(self.info[key])[is_in] for key in self.info}

        return Points(zyx, units=units, resolution=self.resolution, idx=idx, info=info)

    def reset_idx(self):
        """
        Creates new set of idx.
        """
        self.idx = np.arange(self.num_points)


def prepare_shift(shift_by, shift_labels):
    """
    Turns shift_by and shift_labels from config into lists for VolumeDataset.add_shifted_positive_examples()
    :param shift_by:
    :type shift_by:
    :param shift_labels:
    :type shift_labels:
    :return: dict with shift_list, shift_label_list
    :rtype: dict
    """
    # TODO : probably do something smarter in the future.
    #  For example check that there are no synapses at these locations or too close ?
    shift_list = []
    shift_label_list = []
    for shift, label in zip(shift_by, shift_labels):
        shift_list.append([shift, 0, 0])
        shift_list.append([-shift, 0, 0])
        shift_list.append([0, shift, 0])
        shift_list.append([0, -shift, 0])
        shift_list.append([0, 0, shift])
        shift_list.append([0, 0, -shift])

        shift_label_list.extend([label] * 6)

    return {'shifts': shift_list, 'labels': shift_label_list}


def roi_to_centroids(rois):
    """
    creates a centroid for every pixel in the rois
    rois : list of dictionaries, each dict specifies a roi with keys: xmin, xmax, ymin, ymax, zmin, zmax
            list[dict]
    """

    def get_pixels_as_centroids(roi):
        """
        turns pixel into [z,y,x]
        roi: region for which to get the pixels
        """
        z, y, x = np.meshgrid(np.arange(roi['zmin'], roi['zmax'], 1),
                              np.arange(roi['ymin'], roi['ymax'], 1),
                              np.arange(roi['xmin'], roi['xmax'], 1))
        return np.c_[z.flatten(), y.flatten(), x.flatten()]

    centroids = None
    for roi in rois:
        if centroids is None:
            centroids = get_pixels_as_centroids(roi)
        else:
            centroids = np.append(centroids, get_pixels_as_centroids(roi), axis=0)

    return centroids.astype(int)


def get_image_shape(img_file):
    # get file info
    stack = TiffFile(img_file, _multifile=False)
    z_size = len(stack.pages)
    page = stack.pages.get(0)
    y_size, x_size = page.shape
    stack.close()
    return (z_size, y_size, x_size)


def split_to_rois(img_file, zyx_chunks, zyx_padding):
    """
    Splits an image into a given number of rois in z, y, x
    :param img_file: tif file to split
    :param zyx_chunks: list with the number of splits in [z, y, x]
    :param padding: how many pixels to leave uncropped on each side [z, y, x]
    :return: list of dictionaries, each dict specifies a roi with keys: xmin, xmax, ymin, ymax, zmin, zmax
    """

    def get_axis_breaks(axis_size, n, axis_padding):
        """
        Finds the locations on the axis at which to break it
        """
        chunk_size = np.ceil((axis_size - axis_padding * 2) / n).astype(int)
        point = axis_padding
        axis_breaks = []
        while point < (axis_size - axis_padding):
            axis_breaks.append(point)
            point = point + chunk_size
        axis_breaks.append(axis_size - axis_padding)
        return axis_breaks

    (z_size, y_size, x_size) = get_image_shape(img_file)
    # find how to break the axis
    z_breaks = get_axis_breaks(z_size, zyx_chunks[0], zyx_padding[0])
    y_breaks = get_axis_breaks(y_size, zyx_chunks[1], zyx_padding[1])
    x_breaks = get_axis_breaks(x_size, zyx_chunks[2], zyx_padding[2])
    # create the roi dictionaries from breaks
    rois = []
    for iz in range(len(z_breaks) - 1):
        for iy in range(len(y_breaks) - 1):
            for ix in range(len(x_breaks) - 1):
                d = {"zmin": z_breaks[iz], "zmax": z_breaks[iz + 1],
                     "ymin": y_breaks[iy], "ymax": y_breaks[iy + 1],
                     "xmin": x_breaks[ix], "xmax": x_breaks[ix + 1]}
                rois.append(d)
    return rois


def test_numba():
    from loader import load_centroids, load_labels, load_image, drop_unsegmented
    from ugpy.classification.datasets import TwoSlicesDataset
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import torch
    from torch.utils.data import DataLoader, Subset
    import pytorch_lightning as pl

    seed = 222
    pl.seed_everything(seed, workers=True)

    def plot_examples(dataset, main_title):
        print("using dataloader")
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
        fig.suptitle(main_title)
        plt.show()

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

    print("numba = False")
    print("using dataset, getting examples:")
    dataset = TwoSlicesDataset(img, side, centroids, labels=labels, numba=False)
    print(dataset.examples_idx)

    (img1, img2), label = dataset[dataset.examples_idx]
    print(img1.shape)
    print(label)

    plot_examples(dataset, "numba = False")

    print("numba = True")
    print("using dataset, getting examples:")
    dataset = TwoSlicesDataset(img, side, centroids, labels=labels, numba=True)
    print(dataset.examples_idx)

    (img1, img2), label = dataset[dataset.examples_idx]
    print(img1.shape)
    print(label)

    plot_examples(dataset, "numba = True")


if __name__ == "__main__":
    test_numba()
