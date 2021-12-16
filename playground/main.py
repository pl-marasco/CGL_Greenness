import xarray as xr
import rioxarray as riox
import pandas as pd
import numpy as np
import re
import tempfile
import os
import pycurl
import certifi
from distributed import LocalCluster, Client

import functions as fc

from multiprocessing import Pool
from pydap.cas.urs import setup_session
from bs4 import BeautifulSoup
from io import BytesIO


def explorer(url):
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.WRITEDATA, buffer)
    c.setopt(c.CAINFO, certifi.where())
    c.perform()
    c.close()

    body = buffer.getvalue()
    return BeautifulSoup(body.decode('iso-8859-1'), 'html.parser')


def lifter(url, filename):
    if not os.path.isfile(filename):
        # buffer = BytesIO()
        c = pycurl.Curl()

        # c.setopt(c.WRITEDATA, buffer)
        c.setopt(c.CAINFO, certifi.where())
        c.setopt(c.FOLLOWLOCATION, True)
        c.setopt(pycurl.SSL_VERIFYHOST, 2)
        c.setopt(c.NETRC_FILE, r'C:\Users\Pier\_netrc')
        c.setopt(c.NETRC, 1)
        c.setopt(c.COOKIEJAR, r'C:\Users\Pier\_cookie_jar')
        c.setopt(c.URL, url)
        with open(filename, 'wb') as f:
            c.setopt(c.WRITEDATA, f)
            try:
                c.perform()
            except c.error as error:
                # errno, errstr = error
                print(error)
            # c.perform()
            c.close()
            f.close()
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            print(f'Error downloading {filename}')
            return url
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


def download(links):
        archive_folder = r'E:\tmp'

        url_250, url_500 = links

        file_path_250 = file_path_creator(archive_folder, url_250)
        file_path_500 = file_path_creator(archive_folder, url_500)

        failed = []

        fail = lifter(url_500, file_path_500)
        if fail:
            failed.append(fail)

        fail = lifter(url_250, file_path_250)
        if fail:
            failed.append(fail)

        return failed


def main():

    cluster = LocalCluster()
    client = Client(cluster)
    print(client)

    local_folder = r'e:\tmp'

    product_500 = 'MOD09A1'
    product_250 = 'MOD09Q1'

    v_range = range(5, 10)
    h_range = [list(range(17, 24)), list(range(16, 24)), list(range(15, 24)), list(range(21, 24)), list(range(21, 23))]

    tile_list = []
    tile_folder_list = []
    for i, v in enumerate(v_range):
        v = [v]*len(h_range[i])
        pair = list(zip(v, h_range[i]))
        for j in pair:
            folder_500 = os.path.join(product_500, str(j[1]).zfill(2), str(j[0]).zfill(2))
            folder_250 = os.path.join(product_250, str(j[1]).zfill(2), str(j[0]).zfill(2))

            tile_folder_list.append([folder_250, folder_500])

        tile_list.append(pair)

    tile_list = sum(tile_list, [])

    mod250_url = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD09Q1.061/'
    mod500_url = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD09A1.061/'

    soup = explorer(mod250_url)

    products_links = []
    for link in soup.find_all('a')[-5:]:
        soup_tiles_250 = explorer(mod250_url+link.get('href'))
        soup_tiles_500 = explorer(mod500_url+link.get('href'))

        for hv in tile_list:
            v_t = str(hv[0]).zfill(2)
            h_t = str(hv[1]).zfill(2)

            tile_250, tile_500 = ['']*2

            tile_250_list = soup_tiles_250.find_all(href=re.compile(fr'^.*(h{h_t}v{v_t}).*(hdf$)'))
            if tile_250_list:
                tile_250 = tile_250_list[0].getText()

            tile_500_list = soup_tiles_500.find_all(href=re.compile(fr'^.*(h{h_t}v{v_t}).*(hdf$)'))
            if tile_500_list:
                tile_500 = tile_500_list[0].getText()

            if len(tile_250) == len(tile_500) and tile_250:
                products_links.append([mod250_url+link.get('href')+tile_250, mod500_url+link.get('href')+tile_500])

    with Pool(10) as p:
        failed = p.map(download, products_links)

    failed_cln = list(filter(None, failed))
    print(failed_cln)

    gvi_tile = []
    total = len(tile_folder_list)
    t_i = 1
    for tile_folder in tile_folder_list:

        print(f'Processing tile {tile_folder[0][-6:-4]}/{tile_folder[0][-3:-1]} {t_i}/{total}')
        t_i += 1

        filenames_500 = fc.list_hdf_folder(local_folder, tile_folder[1])[-5:]
        filenames_250 = fc.list_hdf_folder(local_folder, tile_folder[0])[-5:]

        if not len(filenames_500) == 5 or not len(filenames_250) == 5:
            print(
                f'Not enought scenes on {tile_folder[0]} [{len(filenames_250)}] or {tile_folder[1]}[{len(filenames_500)}]')
            continue

        filenames = zip(filenames_250, filenames_500)

        scenes = []

        for i in filenames:

            filename_250, filename_500 = i

            with riox.open_rasterio(filename_250,
                                    mask_and_scale=True,
                                    chunks='auto',
                                    lock=False,
                                    variable=['sur_refl_b01', 'sur_refl_b02', 'sur_refl_state_250m']
                                    ) as ds_250:

                Qbits_250 = xr.apply_ufunc(fc._unpackbits, ds_250.sur_refl_state_250m.astype(np.uint16),
                                           kwargs={'num_bits': 16},
                                           input_core_dims=[['y', 'x']],
                                           output_core_dims=[['y', 'x', 'bit']],
                                           vectorize=True,
                                           dask='parallelized',
                                           dask_gufunc_kwargs={'allow_rechunk': True,
                                                               'output_sizes': {'bit': 16}
                                                               }
                                           )
                
                land_mask = np.all(Qbits_250[0, :, :, -6:-3] == np.array([0, 0, 1]), axis=2)
                ds_250 = ds_250.drop_vars('sur_refl_state_250m')
                ds_250_M = ds_250.where(land_mask == True, np.nan)

                doy = filename_250.split(os.sep)[-1].split('_')[1][1:]
                date_250 = pd.to_datetime(f'{doy[:4]}-1-1') + pd.to_timedelta(int(doy[4:]), unit='D')
                ds_250_T = ds_250_M.squeeze().assign_coords({'time': date_250}).expand_dims(dim='time', axis=0).transpose(
                    'time', 'y', 'x').drop('band')
                
            with riox.open_rasterio(filename_500,
                                    mask_and_scale=True,
                                    chunks='auto',
                                    lock=False,
                                    variable=['sur_refl_b06', 'sur_refl_state_500m']) as ds_500:

                Qbits_500 = xr.apply_ufunc(fc._unpackbits, ds_500.sur_refl_state_500m.astype(np.uint16),
                                           kwargs={'num_bits': 16},
                                           input_core_dims=[['y', 'x']],
                                           output_core_dims=[['y', 'x', 'bit']],
                                           vectorize=True,
                                           dask='parallelized',
                                           dask_gufunc_kwargs={'allow_rechunk': True,
                                                               'output_sizes': {'bit': 16}
                                                               }
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

        ndvi = fc._ndvi(ds_scene.sur_refl_b02, ds_scene.sur_refl_b01)
        hsv = xr.apply_ufunc(fc._hsv, ds_scene.sur_refl_b06, ds_scene.sur_refl_b02, ds_scene.sur_refl_b01,
                             input_core_dims=[['y', 'x'], ['y', 'x'], ['y', 'x']],
                             output_core_dims=[['y', 'x', 'hsv']],
                             vectorize=True,
                             dask='parallelized',
                             dask_gufunc_kwargs={'allow_rechunk': True,
                                                 'output_sizes': {'hsv': 3}
                                                 }
                             )

        h = hsv.sel(hsv=0) * 360.

        gvi = xr.apply_ufunc(fc._gvi, ndvi, h,
                             input_core_dims=[['y', 'x'], ['y', 'x']],
                             output_core_dims=[['y', 'x']],
                             vectorize=True,
                             dask='parallelized',
                             dask_gufunc_kwargs={'allow_rechunk': True})

        gvi_rechunked = gvi.chunk({'time': -1})

        gvdm = xr.apply_ufunc(fc._decades, gvi_rechunked,
                              input_core_dims=[['time']],
                              exclude_dims={'time', },
                              dask='parallelized',
                              dask_gufunc_kwargs={'allow_rechunk': True},
                              vectorize=True, )
        gvdm.name = 'greenness'

        gvdm = gvdm.rio.write_crs(ds_merged['spatial_ref'].attrs['crs_wkt'])
        gvdm = gvdm.astype(float)
        gvdm = gvdm.rio.reproject("EPSG:4326", nodata=-999, resampling=0)
        gvdm_nan = gvdm.where(~np.isnan(gvdm), -999)
        gvdm_nodata = gvdm_nan.rio.set_nodata(-999)
        gvdm_f = gvdm_nodata.astype(np.int16)

        gvi_tile.append(gvdm_f)

    print(gvi_tile)


if __name__ == '__main__':
    main()
