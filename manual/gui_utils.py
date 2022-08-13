import numpy as np
import pandas as pd
import json
import csv


def centroids_zx_swap(centroids):
    """Return a copy of centroids array with Z and X swapped, e.g. ZYX<->XYZ."""
    copy = np.zeros(centroids.shape, dtype=centroids.dtype)
    copy[:, 0] = centroids[:, 2]
    copy[:, 1] = centroids[:, 1]
    copy[:, 2] = centroids[:, 0]
    return copy


def load_segment_info_from_csv(infilename, zyx_grid_scale=None, zx_swap=False, filter_status=None):
    """Load a segment list and return content as arrays.

    """
    csvfile = open(infilename, 'r')
    reader = csv.DictReader(csvfile)
    centroids = []
    measures = []
    status = []
    saved_params = None
    for row in reader:
        # newer dump files have an extra saved-parameters row first...
        if row['Z'] == 'saved' and row['Y'] == 'parameters':
            saved_params = row
            continue

        centroids.append(
            (int(row['Z']), int(row['Y']), int(row['X']))
        )
        measures.append(
            (float(row['raw core']), float(row['raw hollow']), float(row['DoG core']), float(row['DoG hollow']))
            + ((float(row['red']),) if 'red' in row else ())
        )
        status.append(
            int(row['override']) if row['override'] else 0
        )
    centroids = np.array(centroids, dtype=np.int32)
    measures = np.array(measures, dtype=np.float32)
    status = np.array(status, dtype=np.uint8)
    if zyx_grid_scale is not None:
        zyx_grid_scale = np.array(zyx_grid_scale, dtype=np.float32)
        assert zyx_grid_scale.shape == (3,)
        centroids = (centroids * zyx_grid_scale).astype(np.float32)
    if filter_status is not None:
        filter_idx = np.zeros(status.shape, dtype=np.bool)
        for value in filter_status:
            filter_idx += (status == value)
        centroids = centroids[filter_idx]
        measures = measures[filter_idx]
        status = status[filter_idx]
    return (
        centroids_zx_swap(centroids) if zx_swap else centroids,
        measures,
        status,
        saved_params
    )


def load_segment_status_from_csv(centroids, offset_origin, infilename):
    """Load a segment list with manual override status values validating against expected centroid list.

       Arguments:
         centroids: Nx3 array of Z,Y,X segment coordinates
         offset_origin: CSV coordinates = offset_origin + centroid coordinates
         infilename: file to open to read CSV content

       Returns tuple with:
         status array (1D),
         saved params dict or None
    """
    csv_centroids, csv_measures, csv_status, saved_params = load_segment_info_from_csv(infilename)
    if csv_centroids.shape[0] > 0:
        csv_centroids -= np.array(offset_origin, dtype=np.int32)
    return dense_segment_status(centroids, csv_centroids, csv_status), saved_params


def dense_segment_status(centroids, sparse_centroids, sparse_status):
    """Construct dense segment status from sparse info, e.g. previously loaded from CSV."""
    # assume that dump is ordered subset of current analysis
    status = np.zeros((centroids.shape[0],), dtype=np.uint8)

    i = 0
    for row in range(sparse_centroids.shape[0]):
        # scan forward until we find same centroid in sparse subset
        while i < centroids.shape[0] and tuple(sparse_centroids[row]) != tuple(centroids[i]):
            i += 1

        if i >= centroids.shape[0]:
            raise ValueError("Sparse dump does not match image analysis!", sparse_centroids[row])

        if sparse_status[row]:
            status[i] = sparse_status[row]

    return status


def get_centroids_and_labels(npz_file, csv_file=None):
    """
    Extracts potential synapse centroids from the npz.

    Parameters:
        npz_file: path to the saved npz file
        csv_file: path to the saved csv file (optional)
    Returns:
        centroids: centroid coordinates in the Tiff space, in pixels

    """

    def get_centroidsfrom_parts(parts):
        centroids = parts['centroids'].astype(np.int32)
        props = json.loads(parts['properties'].tostring().decode('utf8'))
        slice_origin = np.array(props['slice_origin'], dtype=np.int32)

        return centroids, slice_origin

    parts = np.load(npz_file)
    centroids, slice_origin = get_centroidsfrom_parts(parts)

    if csv_file:
        # build dense status array from sparse CSV
        statuses, _ = load_segment_status_from_csv(centroids, slice_origin, csv_file)
        # interpret status flag values
        is_synapse = statuses == 7
        labels = is_synapse.astype(int)
    else:
        # set all to " not synapse "
        labels = centroids[:, 0] * 0

    # convert cropped centroids back to full SPIM voxel coords
    centroids_tiff = centroids + slice_origin

    return centroids_tiff, labels


def export_predictions_gui(npz_file, export_path, csv_file=None):
    """
    Saves predictions as a file that can be used with the segmentation gui.
    """
    # TODO : make for more rids
    # rid = cnn_data.rid[0]
    npz = np.load(npz_file)
    measures = npz['measures']
    centroids, labels = get_centroids_and_labels(npz_file, csv_file=csv_file)
    gui_df = pd.DataFrame({'X': np.squeeze(centroids[:, 2]),
                           'Y': np.squeeze(centroids[:, 1]),
                           'Z': np.squeeze(centroids[:, 0]),
                           'prob': np.squeeze(labels),
                           'm1': np.squeeze(measures[:, 0]),
                           'm2': np.squeeze(measures[:, 1]),
                           'm3': np.squeeze(measures[:, 2]),
                           'm4': np.squeeze(measures[:, 3])})
    gui_df.to_csv(export_path, index=False)
