import glob
import os

import numpy as np
from skimage.color import rgb2hsv


def _ndvi(nir, red):
    da = (nir-red)/(nir+red)
    da.name = 'NDVI'
    return da


def _hsv(R, G, B):
    rgb = np.dstack((R*255, G*255, B*255))
    hsv = rgb2hsv(rgb)
    hsv_nan = np.where(hsv != 0, hsv, np.nan)
    return hsv_nan


def _gvi(ndvi, h):
    null_mask = np.logical_or(np.isnan(ndvi), np.isnan(h))

    vegetated = np.where((h >= (-2354.83 * ndvi) + 522.68), 1, 0)
    semiveg = np.where(
        (h > (-2139.54 * ndvi) + 377.63) & (h < (57.22 * ndvi) + 141.42) & (h < (-2354.83 * ndvi) + 522.68) & (
                    h > (-261.64 * ndvi) + 133.30), 1, 0)

    # slope = _slope(HSV_d)
    # semi_vegetated = HSV_d.where(semiveg) #.where(slope > 11.9)

    #     gvi = np.logical_or(~np.isnan(vegetated), ~np.isnan(semiveg))
    gvi = np.logical_or(vegetated, semiveg)

    gvi_masked = np.where(~null_mask, gvi, np.NaN)

    return gvi_masked


def _decades(data):
    if data[-1] == 1:
        diff = np.diff(data)
        # return np.split(data, np.where(np.diff(data) != 0)[0]+1)[-1].size
        return np.split(data, np.where(np.logical_and(~np.isnan(diff), diff != 0))[0]+1)[-1].size
    else:
        return -999


def _unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return np.fliplr((x & mask).astype(bool).astype(int)).reshape(xshape + [num_bits])


# def list_blobs_in_folder(container_name, folder_name):
#     """
#     List all blobs in a virtual folder in an Azure blob container
#     """
#
#     files = []
#     generator = modis_container_client.list_blobs(name_starts_with=folder_name)
#     for blob in generator:
#         files.append(blob.name)
#     return files


def list_hdf_folder(root_name, folders_path):
    """"
    List .hdf files in a folder
    """

    query_str = os.path.join(root_name, folders_path, '*.hdf')
    files = glob.glob(query_str)
    return files