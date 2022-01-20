import datetime

import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
import re
import os
import pycurl
import certifi
import glob
import os
import dask
import platform

from rioxarray import merge
from distributed import LocalCluster, Client, wait
from distributed.utils import tmpfile
from multiprocessing import Pool, Process
# from pydap.cas.urs import setup_session
from tqdm.contrib.concurrent import thread_map
from dask_jobqueue import PBSCluster
from skimage.color import rgb2hsv
from bs4 import BeautifulSoup
from io import BytesIO


def _ndvi(red, nir):
    da = (nir - red) / (nir + red)
    da.name = 'NDVI'
    return da


def _evi(blue, red, nir):
    2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))


def _rescale(in_array, input_low, input_high, out_low=0, out_high=1):
    return ((in_array - input_low)/(input_high-input_low))*(out_high-out_low)+out_low


def _hsv(R, G, B):
    r = _rescale(R, -0.01, 1.6, 0, 255)
    g = _rescale(G, -0.01, 1.6, 0, 255)
    b = _rescale(B, -0.01, 1.6, 0, 255)

    rgb = np.dstack((r, g, b))
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
        return np.split(data, np.where(np.logical_and(~np.isnan(diff), diff != 0))[0] + 1)[-1].size
    else:
        return -999


def _unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
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


def scraping(url):
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    c.setopt(c.CAINFO, certifi.where())
    c.perform()
    c.close()

    body = buffer.getvalue()
    return BeautifulSoup(body.decode('iso-8859-1'), 'html.parser')


def xml_size_extractor(file_path):

    with open(file_path, 'rb') as body:
        xml_body = BeautifulSoup(body.read().decode('UTF-8'), 'lxml')

    return int(xml_body.find('filesize').text)


def download(url, filename, netrc_path=r'/home/maraspi/.netrc', cookie_path=r'/home/maraspi/.cookie_jar'):

    c = pycurl.Curl()

    c.setopt(c.CAINFO, certifi.where())
    c.setopt(c.FOLLOWLOCATION, True)
    c.setopt(pycurl.SSL_VERIFYHOST, 2)
    c.setopt(c.NETRC_FILE, netrc_path)
    c.setopt(c.NETRC, 1)
    c.setopt(c.COOKIEJAR, cookie_path)
    c.setopt(c.NOSIGNAL, 1)
    c.setopt(c.URL, url)

    with open(filename, 'wb') as f:
        c.setopt(c.WRITEDATA, f)
        try:
            c.perform()
        except c.error as error:
            raise error

        c.close()
        f.close()
    return


def file_path_creator(archive_folder, url):
    file_nm = url.split('/')[-1]
    file_nm_components = file_nm.split('.')

    h = file_nm_components[2][1:3]
    v = file_nm_components[2][4:6]
    product_nm = file_nm_components[0]

    folder = os.path.join(archive_folder, product_nm, h, v)
    os.makedirs(folder, exist_ok=True)

    tile_nm = file_nm.replace('.', '_').replace('_hdf', '.hdf')

    file_path = os.path.join(folder, tile_nm)

    return file_path


def v_h_extractor(path):
    components = path[0].split(os.sep)
    v = components[-1]
    h = components[-2]
    return h, v


def file_withdraw(pack):
    archive_folder, netrc_path, cookie_path = pack[1]

    url = pack[0]

    file_path = file_path_creator(archive_folder, url)
    xml_file_path = file_path + '.xml'

    if os.path.isfile(xml_file_path):
        verbatim_size = xml_size_extractor(xml_file_path)
    else:
        i = 0
        xml_url = url + '.xml'
        while i <= 3:
            try:
                download(xml_url, xml_file_path, netrc_path, cookie_path)
            except:
                i += 1
            else:
                break
        if i == 2:
            print(f'Error downloading {xml_url}, 3 attempts have been done. Please check manually')

        verbatim_size = xml_size_extractor(xml_file_path)

    if not os.path.isfile(file_path) or os.path.getsize(file_path) != verbatim_size:
        i = 0
        while i <= 3:
            try:
                download(url, file_path, netrc_path, cookie_path)
            except:
                print(f'Error downloading {url}')

            if os.path.getsize(file_path) != verbatim_size:
                print(f'Size error in {file_path}')
                i += 1
            else:
                break
        if os.path.getsize(file_path) != verbatim_size:
            print(f'Multiple attempt of retreating file {url} have been conducted, please check manually')
            return url

    return


@dask.delayed
def greenness_detection(tile_folders, local_folder):
    h_num, v_num = v_h_extractor(tile_folders)

    filenames_500 = list_hdf_folder(local_folder, tile_folders[1])[-5:]
    filenames_250 = list_hdf_folder(local_folder, tile_folders[0])[-5:]

    if not len(filenames_500) == 5 or not len(filenames_250) == 5:
        print(
            f'Not enought scenes on {tile_folders[0]} [{len(filenames_250)}] or {tile_folders[1]}[{len(filenames_500)}]')
        return

    filenames = zip(filenames_250, filenames_500)

    scenes = []

    for i in filenames:
        filename_250, filename_500 = i

        with rioxarray.open_rasterio(filename_250,
                                     mask_and_scale=True,
                                     chunks='auto',
                                     lock=False,
                                     variable=['sur_refl_b01', 'sur_refl_b02', 'sur_refl_state_250m']
                                     ) as ds_250:
            Qbits_250 = xr.apply_ufunc(_unpackbits, ds_250.sur_refl_state_250m.astype(np.uint16),
                                       kwargs={'num_bits': 16},
                                       input_core_dims=[['y', 'x']],
                                       output_core_dims=[['y', 'x', 'bit']],
                                       vectorize=True,
                                       dask='parallelized',
                                       dask_gufunc_kwargs={'allow_rechunk': True,
                                                           'output_sizes': {'bit': 16}}
                                       )

            land_mask = np.all(Qbits_250[0, :, :, -6:-3] == np.array([0, 0, 1]), axis=2)
            ds_250 = ds_250.drop_vars('sur_refl_state_250m')
            ds_250_M = ds_250.where(land_mask == True, np.nan)

            doy = filename_250.split(os.sep)[-1].split('_')[1][1:]
            date_250 = pd.to_datetime(f'{doy[:4]}-1-1') + pd.to_timedelta(int(doy[4:]), unit='D')
            ds_250_T = ds_250_M.squeeze().assign_coords({'time': date_250}).expand_dims(dim='time', axis=0).transpose(
                'time', 'y', 'x').drop('band')

        with rioxarray.open_rasterio(filename_500,
                                     mask_and_scale=True,
                                     chunks='auto',
                                     lock=False,
                                     variable=['sur_refl_b03', 'sur_refl_b06', 'sur_refl_state_500m']) as ds_500:
            Qbits_500 = xr.apply_ufunc(_unpackbits, ds_500.sur_refl_state_500m.astype(np.uint16),
                                       kwargs={'num_bits': 16},
                                       input_core_dims=[['y', 'x']],
                                       output_core_dims=[['y', 'x', 'bit']],
                                       vectorize=True,
                                       dask='parallelized',
                                       dask_gufunc_kwargs={'allow_rechunk': True,
                                                           'output_sizes': {'bit': 16}}
                                       )

            land_mask_500 = np.all(Qbits_500[0, :, :, -6:-3] == np.array([0, 0, 1]), axis=2)
            ds_500 = ds_500.drop_vars('sur_refl_state_500m')
            ds_500_M = ds_500.where(land_mask_500 == True, np.nan)

            ds_500_match = ds_500_M.rio.reproject_match(ds_250).assign_coords({'x': ds_250.x, 'y': ds_250.y})

            doy_500 = filename_500.split(os.sep)[-1].split('_')[1][1:]
            date_500 = pd.to_datetime(f'{doy_500[:4]}-1-1') + pd.to_timedelta(int(doy_500[4:]), unit='D')

            ds_500_T = ds_500_match.squeeze().assign_coords({'time': date_500}).expand_dims(dim='time',
                                                                                            axis=0).transpose('time',
                                                                                                              'y',
                                                                                                              'x').drop(
                'band')

            ds_500_C = ds_500_T.chunk(ds_250_T.chunks)

        ds_merged = xr.merge([ds_250_T, ds_500_C])
        scenes.append(ds_merged)

    # Create a DataSet per tile and rechunk on time dimension
    ds_scene = xr.merge(scenes)
    ds_scene = ds_scene.chunk({'time': 1, 'y': 'auto', 'x': 'auto'})

    ndvi = _ndvi(ds_scene.sur_refl_b01, ds_scene.sur_refl_b02)
    ndvi = ndvi.where(ndvi <= 1.0, 1.0)

    #evi = _evi(ds_scene.sur_refl_b03,  ds_scene.sur_refl_b02, ds_scene.sur_refl_b01 )

    hsv = xr.apply_ufunc(_hsv, ds_scene.sur_refl_b06, ds_scene.sur_refl_b02, ds_scene.sur_refl_b01,
                         input_core_dims=[['y', 'x'], ['y', 'x'], ['y', 'x']],
                         output_core_dims=[['y', 'x', 'hsv']],
                         vectorize=True,
                         dask='parallelized',
                         dask_gufunc_kwargs={'allow_rechunk': True,
                                             'output_sizes': {'hsv': 3}}
                         )

    h = hsv.sel(hsv=0) * 360.

    gvi = xr.apply_ufunc(_gvi, ndvi, h,
                         input_core_dims=[['y', 'x'], ['y', 'x']],
                         output_core_dims=[['y', 'x']],
                         vectorize=True,
                         dask='parallelized',
                         dask_gufunc_kwargs={'allow_rechunk': True}
                         )

    gvi_rechunked = gvi.chunk({'time': -1})

    gvdm = xr.apply_ufunc(_decades, gvi_rechunked,
                          input_core_dims=[['time']],
                          exclude_dims={'time', },
                          dask='parallelized',
                          dask_gufunc_kwargs={'allow_rechunk': True},
                          vectorize=True, )
    gvdm.name = 'greenness'

    gvdm = gvdm.rio.write_crs(ds_merged['spatial_ref'].attrs['crs_wkt']).astype(float)

    gvdm = gvdm.rio.reproject("EPSG:4326", nodata=-999, resampling=0)
    gvdm_nan = gvdm.where(~np.isnan(gvdm), -999)
    gvdm_nodata = gvdm_nan.rio.set_nodata(-999)
    gvdm_f = gvdm_nodata.astype(np.int16)

    # region NDVI
    ndvi_ = ndvi.rio.write_crs(ds_merged['spatial_ref'].attrs['crs_wkt']).astype(float)
    ndvi_out = ndvi_.rio.reproject("EPSG:4326", nodata=-999, resampling=0)
    tile_name = f'ndvi_{h_num}_{v_num}.nc'
    ndvi_out_path = os.path.join(local_folder, 'Results', tile_name)
    ndvi_out.to_netcdf(ndvi_out_path)
    # endregion

    # region H
    h_ = h.rio.write_crs(ds_merged['spatial_ref'].attrs['crs_wkt']).astype(float)
    h_out = h_.rio.reproject("EPSG:4326", nodata=-999, resampling=0)
    h_tile_name = f'h_{h_num}_{v_num}.nc'
    h_out_path = os.path.join(local_folder, 'Results', h_tile_name)
    h_out.to_netcdf(h_out_path)
    # endregion

    # region GVI
    gvi_ = gvi.rio.write_crs(ds_merged['spatial_ref'].attrs['crs_wkt']).astype(float)
    gvi_out = gvi_.rio.reproject("EPSG:4326", nodata=-999, resampling=0)
    gvi_tile_name = f'gvi_{h_num}_{v_num}.nc'
    gvi_out_path = os.path.join(local_folder, 'Results', gvi_tile_name)
    gvi_out.to_netcdf(gvi_out_path)
    # endregion

    return gvdm_f


def main():
    env = platform.system()
    today_date = datetime.date.today().strftime('%Y%m%d')

    if env == 'Windows':
        # local_folder = r'c:\temp'
        local_folder = r'e:\tmp'
        netrc_path = r'c:\Users\Pier\.netrc'
        cookie_path = r'c:\Users\Pier\.cookie_jar'
        # out_path = r'c:\temp\Results\out.tif'
        out_path = fr'e:\tmp\results\MVP_Modis_250_{today_date}.tif'

        options = [local_folder, netrc_path, cookie_path]

        workers = 4
    else:

        local_folder = r'/BGFS/COMMON/maraspi/Modis'
        netrc_path = r'/home/maraspi/.netrc'
        cookie_path = r'/home/maraspi/.cookie_jar'
        out_path = fr'/BGFS/COMMON/maraspi/Modis/Results/MVP_Modis_250_{today_date}.tif'

        options = [local_folder, netrc_path, cookie_path]

        cluster = PBSCluster(cores=32,
                             processes=4,
                             # memory="240GB",
                             project='DASK_Parabellum',
                             queue='high',
                             # local_directory='/local0/maraspi/',
                             walltime='12:00:00',
                             # death_timeout=240,
                             log_directory='/tmp/marapi/workers/')

        workers = 32

    product_500 = 'MOD09A1'
    product_250 = 'MOD09Q1'

    v_range = range(5, 10)
    h_range = [list(range(17, 24)), list(range(16, 27)), list(range(15, 27)), list(range(20, 24)), list(range(20, 23))]

    # v_range = range(7, 8)
    # h_range = [range(16, 17), ]

    tile_list = []
    tile_folder_list = []
    for i, v in enumerate(v_range):
        v = [v] * len(h_range[i])
        pair = list(zip(v, h_range[i]))
        for j in pair:
            folder_500 = os.path.join(product_500, str(j[1]).zfill(2), str(j[0]).zfill(2))
            folder_250 = os.path.join(product_250, str(j[1]).zfill(2), str(j[0]).zfill(2))

            tile_folder_list.append([folder_250, folder_500])

        tile_list.append(pair)

    tile_list = sum(tile_list, [])
    print('Tile list completed')

    mod250_url = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD09Q1.061/'
    mod500_url = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD09A1.061/'

    soup = scraping(mod250_url)
    print('Modis date list retreated')

    products_links = []
    for date_link in soup.find_all('a')[-5:]:
        date_str = date_link.get('href')
        soup_tiles_250 = scraping(mod250_url + date_str)
        soup_tiles_500 = scraping(mod500_url + date_str)

        for hv in tile_list:
            v_t = str(hv[0]).zfill(2)
            h_t = str(hv[1]).zfill(2)

            tile_250_list = soup_tiles_250.find_all(href=re.compile(fr'^.*(h{h_t}v{v_t}).*(hdf)'))
            if tile_250_list:
                tile_250 = tile_250_list[0].getText()
                products_links.append(mod250_url + date_link.get('href') + tile_250)

            tile_500_list = soup_tiles_500.find_all(href=re.compile(fr'^.*(h{h_t}v{v_t}).*(hdf)'))
            if tile_500_list:
                tile_500 = tile_500_list[0].getText()
                products_links.append(mod500_url + date_link.get('href') + tile_500)

    print('Product link created')

    # with Pool(5, maxtasksperchild=None) as p:
    #     failed = p.map(download, zip(products_links, [options] * len(products_links)), chunksize=1)

    failed = thread_map(file_withdraw, zip(products_links, [options] * len(products_links)), total=len(products_links))

    # for product in products_links:
    #     failed = file_withdraw([product, options])
    #

    print('Products downloaded')

    failed_cln = list(filter(None, failed))
    if failed_cln:
        print(failed_cln)

    if env == 'Windows':
        cluster = LocalCluster(n_workers=workers)
        client = Client(cluster)
        client.wait_for_workers(workers)
    else:
        cluster.scale(workers)
        client = Client(cluster)
        client.wait_for_workers((workers * 4)/64)

    print(client)

    d_tiles = [greenness_detection(tile_folder, local_folder) for tile_folder in tile_folder_list]

    # print(d_tiles)
    GVI_tiles = dask.compute(d_tiles)

    print('Merge')

    GVI = rioxarray.merge.merge_arrays(GVI_tiles[0])

    # GVI = rioxarray.merge.merge_arrays(d_tiles)

    print('out')
    GVI.rio.to_raster(out_path, **{'compress': 'lzw'})


if __name__ == '__main__':
    main()
