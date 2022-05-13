import argparse
import asyncio
import datetime
import gc
import glob
import os
import platform
import sys
import asyncssh
import numpy as np
import pandas as pd
import xarray as xr
# import rioxarray

from urllib.parse import urljoin
from dask_jobqueue import PBSCluster
from distributed import Client, LocalCluster
import dask
from numba import jit
from skimage.color import rgb2hsv

np.seterr(all='ignore')


class ProcessSettings:

    def __init__(s, AOI, bando, pxl_sz, grid_path, s_date, time_delta,
                 local_folder, out_path, out_name, archive_path, flush,
                 server, port, user, password, root_path):

        # time
        s.start_ref_date = s.__10D_ref_date_finder(s_date)
        s.end_ref_date = s.__10D_ref_date_finder(s.start_ref_date - datetime.timedelta(time_delta))
        s.date_range = s.__date_range(s.start_ref_date, s.end_ref_date)
        s.D10_range = s.__D10_range_create(s.date_range)
        s.D10_required = None

        # space
        s.AOI_TL_x = AOI[0]
        s.AOI_TL_y = AOI[1]
        s.AOI_BR_x = AOI[2]
        s.AOI_BR_y = AOI[3]

        s.AOI_TL_str = s.__tile(AOI[0], AOI[1])
        s.AOI_BR_str = s.__tile(AOI[2], AOI[3])
        s.x_range = s.__range(AOI[0], AOI[2])
        s.y_range = s.__range(AOI[1], AOI[3])
        s.tile_list = s.__tile_list(s.x_range, s.y_range, bando)
        s.grid = s._grid_creator(-180., -65., +180., +85., (1 / 336), 3360, 3360)

        s.minx = s.grid[s.AOI_TL_str].bounds[0]
        s.maxy = s.grid[s.AOI_TL_str].bounds[3]
        s.miny = s.grid[s.AOI_BR_str].bounds[1]
        s.maxx = s.grid[s.AOI_BR_str].bounds[2]

        s.lat_n = int(((s.maxy - s.miny) / pxl_sz))
        s.lon_n = int(((s.maxx - s.minx) / pxl_sz))

        # local paths
        s.local_folder = local_folder
        s.out_path = out_path
        s.out_name = f'{out_name}_{str(s.D10_range[-2].year)}{str(s.D10_range[-2].month).zfill(2)}{str(s.D10_range[-2].day).zfill(2)}.tif'

        s.results_path = os.path.join(s.out_path, s.out_name)
        s.archive_path = archive_path
        s.flush = flush

        # server info
        s.server = server
        s.port = port
        s.user = user
        s.password = password
        s.root_path = root_path

    def _grid_creator(s, xmin, ymin, xmax, ymax, px_size, x_tile_dim, y_tile_dim):
        from shapely.geometry import Polygon

        wide = px_size * x_tile_dim
        length = px_size * y_tile_dim

        cols = list(np.arange(xmin, xmax + wide, wide))
        rows = list(np.arange(ymax, ymin - length, -length))

        grid = {}
        for xi, x in enumerate(cols[:-1]):
            for yi, y in enumerate(rows[:-1]):
                grid[f'X{str(xi).zfill(2)}Y{str(yi).zfill(2)}'] = Polygon(
                    [(x, y), (x + wide, y), (x + wide, y - length), (x, y - length)])
        return grid

    def __tile(self, x, y):
        return f'X{str(x).zfill(2)}Y{str(y).zfill(2)}'

    def __range(self, min, max):
        return range(min, max + 1)

    def __10D_ref_date_finder(self, date):
        if date.day == 1:
            if date.month == 1:
                return datetime.datetime(date.year-1, 12, 21)
            return datetime.datetime(date.year, date.month - 1, 21)
        elif 1 < date.day <= 11:
            ref_day = 1
        elif 11 < date.day <= 21:
            ref_day = 11
        else:
            ref_day = 21
        return datetime.datetime(date.year, date.month, ref_day)

    def __date_range(self, start, end):
        return pd.date_range(end, start, freq='D', inclusive='both')

    def __tile_list(self, x_range, y_range, bando):
        tile_list = []
        for i, v in enumerate(y_range):
            v = [v] * len(x_range)

            tile_list.append(list(map(lambda j: fr'X{str(j[1]).zfill(2)}Y{str(j[0]).zfill(2)}', zip(v, x_range))))

        full = sum(tile_list, [])

        if bando:
            [full.remove(i) for i in bando if i in full]

        return full

    def __D10_range_create(self, date_range):
        return date_range[date_range.day.isin([1, 11, 21])]


class DownloadError:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.error = args
        print(self.error)


class ProgressHandler:
    def __init__(self, task_id, *args, **kwargs):
        self.id = task_id

    def __call__(self, remote_folder, local_folder, dwn_size, total_size, *args, **kwargs):
        if dwn_size == total_size:
            print(
                f'Worker {self.id} downloaded {os.path.basename(local_folder).decode()} for a total of {round(total_size * 0.000000953674316, 1)} MB',
                flush=True)


# region Managment
async def is_local(file, tile_list, local_path, yr, mm, str_date):
    if file.filename in ['.', '..', 'manifest.txt']:
        return True

    if file.filename.split('_')[3] in tile_list:

        file_path = os.path.join(local_path, 'archive', yr, mm, str_date, file.filename)

        if os.path.isfile(file_path) and os.path.getsize(file_path) == file.attrs.size:
            return True
        else:
            return False
    else:
        return True


def split_date(date: datetime.date) -> (str, str, str):
    yr = date.year.__str__()
    mm = date.month.__str__().zfill(2)
    str_date = date.strftime('%Y%m%d')
    return yr, mm, str_date


async def get_files(server, port, user, password, root_path, local_path, queue):
    task_name = asyncio.current_task().get_name()

    async with asyncssh.connect(server, port, username=user, password=password, known_hosts=None) as conn:
        async with conn.start_sftp_client() as sftp:
            while True:
                progress_handler = ProgressHandler(task_name)

                sel_date, tiles = await queue.get()

                yr, mm, str_date = split_date(sel_date)

                spef_local_pth = os.path.join(local_path, 'archive', yr, mm, str_date)
                os.makedirs(spef_local_pth, exist_ok=True)

                abs_dir_path = urljoin(root_path, f'{yr}/{mm}/{str_date}/')

                dwnl_file_list = []
                try:
                    async for file in sftp.scandir(abs_dir_path):
                        if not await is_local(file, tiles, local_path, yr, mm, str_date):
                            dwnl_file_list.append(urljoin(abs_dir_path, file.filename))
                except(asyncssh.Error, OSError) as e:
                    raise e

                try:
                    if len(dwnl_file_list) != 0:
                        await sftp.get(dwnl_file_list, spef_local_pth, max_requests=128, error_handler=DownloadError,
                                       progress_handler=progress_handler, recurse=True)
                        print(
                            f'Worker {task_name} got all files for the {sel_date.year}-{sel_date.month.__str__().zfill(2)}-{sel_date.day.__str__().zfill(2)}')
                        queue.task_done()
                    else:
                        print(
                            f'Worker {task_name} verified that all files for {sel_date.year}-{sel_date.month.__str__().zfill(2)}-{sel_date.day.__str__().zfill(2)} are already in house')
                        queue.task_done()
                except (asyncssh.Error, OSError) as e:
                    raise e


def archive_creator(archive_path, D10_dates, lat_n, lon_n, minx, maxx, maxy, miny):
    time_n = D10_dates[:-1].size

    empty_array = dask.array.empty(shape=(time_n, lat_n, lon_n), chunks=(1, 3360, 3360))
    zero_array = dask.array.zeros_like(empty_array)

    lon = np.round(np.linspace(minx, maxx, lon_n, endpoint=False), 8).squeeze()
    lat = np.round(np.linspace(maxy, miny, lat_n, endpoint=False), 8).squeeze()

    gvi = xr.DataArray(data=zero_array, coords={'time': D10_dates[:-1], 'lat': lat.T, 'lon': lon.T, }, name='GVI')

    gvi.to_dataset().to_zarr(archive_path, compute=False, consolidated=True, mode='w')

    return D10_dates[:-1]


def archive_append(archive_path, D10_dates):
    archive = xr.open_dataset(archive_path, engine='zarr', consolidated=True)

    lat = archive.lat
    lon = archive.lon

    alignment = D10_dates[:-1][~D10_dates[:-1].isin(archive.time.values)]

    if not alignment.empty:
        time_n = alignment.size

        empty_array = dask.array.empty(shape=(time_n, lat.size, lon.size), chunks=(1, 3360, 3360))
        zero_array = dask.array.zeros_like(empty_array)

        gvi = xr.DataArray(data=zero_array, coords={'time': alignment, 'lat': lat.T, 'lon': lon.T},
                           name='GVI').to_dataset()

        gvi.to_zarr(archive_path, append_dim='time', consolidated=True)

    return alignment


async def lifter(s):
    tile2download = await download_list(s.D10_required, s.tile_list)

    tasks = asyncio.Queue(len(tile2download))
    for item in tile2download:
        tasks.put_nowait(item)

    workers = [asyncio.create_task(
        get_files(s.server, s.port, s.user, s.password, s.root_path, s.local_folder, tasks), name=str(i)) for i in
        range(8)]

    await tasks.join()

    for worker in workers:
        worker.cancel()

    results = await asyncio.gather(*workers, return_exceptions=True)

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print('\rTask %d failed: %s' % (i, str(result)))
        else:
            print('\rTask %d succeeded:' % i)


async def download_list(D10_list, tile_range):
    if D10_list[-1].day == 1:
        D10_end = datetime.datetime.strptime(f'{D10_list[-1].year}{D10_list[-1].month.__str__().zfill(2)}11',
                                             '%Y%m%d').date()
    elif D10_list[-1].day == 11:
        D10_end = datetime.datetime.strptime(f'{D10_list[-1].year}{D10_list[-1].month.__str__().zfill(2)}21',
                                             '%Y%m%d').date()
    else:
        if D10_list[-1].month == 12:
            D10_end = datetime.datetime.strptime(f'{D10_list[-1].year + 1}0101', '%Y%m%d').date()
        else:
            D10_end = datetime.datetime.strptime(f'{D10_list[-1].year}{(D10_list[-1].month + 1).__str__().zfill(2)}01',
                                                 '%Y%m%d').date()

    date_range = pd.date_range(D10_list[0], D10_end, freq='D', inclusive='left')
    return list(zip(date_range, [tile_range] * date_range.size))


def ds_opener(band_path):
    ds = xr.open_dataset(band_path, cache=True)

    pixel_classif_flags = xr.apply_ufunc(_unpackbits, ds.pixel_classif_flags.astype(np.uint32),
                                         kwargs={'num_bits': 10},
                                         input_core_dims=[['lat', 'lon']],
                                         output_core_dims=[['lat', 'lon', 'bit']],
                                         vectorize=True,
                                         dask='parallelized',
                                         dask_gufunc_kwargs={'allow_rechunk': True,
                                                             'output_sizes': {'bit': 10}}
                                         )
    quality_flags = xr.apply_ufunc(_unpackbits, ds.quality_flags.astype(np.uint32),
                                   kwargs={'num_bits': 32},
                                   input_core_dims=[['lat', 'lon']],
                                   output_core_dims=[['lat', 'lon', 'bit']],
                                   vectorize=True,
                                   dask='parallelized',
                                   dask_gufunc_kwargs={'allow_rechunk': True,
                                                       'output_sizes': {'bit': 32}}
                                   )

    cloud_an = xr.apply_ufunc(_unpackbits, ds.cloud_an.astype(np.uint32),
                                   kwargs={'num_bits': 14},
                                   input_core_dims=[['lat', 'lon']],
                                   output_core_dims=[['lat', 'lon', 'bit']],
                                   vectorize=True,
                                   dask='parallelized',
                                   dask_gufunc_kwargs={'allow_rechunk': True,
                                                       'output_sizes': {'bit': 14}}
                                   )

    AC_process = xr.apply_ufunc(_unpackbits, ds.AC_process_flag.astype(np.uint32),
                                   kwargs={'num_bits': 6},
                                   input_core_dims=[['lat', 'lon']],
                                   output_core_dims=[['lat', 'lon', 'bit']],
                                   vectorize=True,
                                   dask='parallelized',
                                   dask_gufunc_kwargs={'allow_rechunk': True,
                                                       'output_sizes': {'bit': 6}}
                                   )

    Q_mask = xr.where(~((pixel_classif_flags[:, :, 0] == 1) |
                        (pixel_classif_flags[:, :, 1] == 1) |
                        (pixel_classif_flags[:, :, 2] == 1) |
                        (pixel_classif_flags[:, :, 3] == 1) |
                        (pixel_classif_flags[:, :, 4] == 1) |
                        (pixel_classif_flags[:, :, 5] == 1) |
                        (pixel_classif_flags[:, :, 6] == 1) |
                        (pixel_classif_flags[:, :, 7] == 1) |
                        (pixel_classif_flags[:, :, 8] == 1) |

                        (quality_flags[:, :, 3] == 1) |
                        (quality_flags[:, :, 4] == 1) |
                        (quality_flags[:, :, 5] == 1) |
                        (quality_flags[:, :, 6] == 1) |
                        (quality_flags[:, :, 7] == 1) |
                        (quality_flags[:, :, 8] == 1) |
                        (quality_flags[:, :, 9] == 1) |
                        (quality_flags[:, :, 10] == 1) |
                        (quality_flags[:, :, 16] == 1) |
                        (quality_flags[:, :, 17] == 1) |
                        (quality_flags[:, :, 18] == 1) |
                        (quality_flags[:, :, 23] == 1) |
                        (quality_flags[:, :, 24] == 1) |
                        (quality_flags[:, :, 25] == 1) |
                        # (quality_flags[:, :, 31] == 0) |

                        (cloud_an[:, :, 0] == 1) |
                        (cloud_an[:, :, 1] == 1) |
                        (cloud_an[:, :, 2] == 1) |
                        (cloud_an[:, :, 3] == 1) |
                        (cloud_an[:, :, 4] == 1) |
                        (cloud_an[:, :, 5] == 1) |
                        (cloud_an[:, :, 6] == 1) |
                        (cloud_an[:, :, 7] == 1) |
                        (cloud_an[:, :, 8] == 1) |
                        (cloud_an[:, :, 9] == 1) |
                        (cloud_an[:, :, 10] == 1) |
                        (cloud_an[:, :, 11] == 1) |
                        (cloud_an[:, :, 12] == 1) |
                        (cloud_an[:, :, 13] == 1) |

                        ((AC_process[:, :, 2] == 1) & (AC_process[:, :, 1] == 0)) |
                        ((AC_process[:, :, 2] == 1) & (AC_process[:, :, 1] == 1)) |
                        (AC_process[:, :, 3] == 1) |

                        (ds.VZA_olci > 50) |
                        (ds.VZA_slstr > 50)
                        ), True, False)

    ds = ds.drop_vars(['Oa02_toc', 'Oa02_toc_error',
                       'Oa03_toc_error',
                       'Oa04_toc_error',
                       'Oa05_toc_error',
                       'Oa06_toc_error',
                       'Oa11_toc', 'Oa11_toc_error',
                       'Oa12_toc', 'Oa12_toc_error',
                       'Oa21_toc', 'Oa21_toc_error',
                       'S1_an_toc', 'S1_an_toc_error',
                       'S2_an_toc', 'S2_an_toc_error',
                       'S3_an_toc', 'S3_an_toc_error',
                       'S6_an_toc', 'S6_an_toc_error',
                       'Oa07_toc_error',
                       'Oa08_toc_error',
                       'Oa09_toc_error',
                       'Oa10_toc_error',
                       'Oa16_toc_error',
                       'Oa17_toc_error',
                       'Oa18_toc_error',
                       'S5_an_toc_error',
                       'SAA_olci',
                       'SZA_olci',
                       'VAA_olci',
                       'VZA_olci',
                       'SAA_slstr',
                       'SZA_slstr',
                       'VAA_slstr',
                       'VZA_slstr',
                       'cloud_an',
                       'quality_flags',
                       'pixel_classif_flags',
                       'AC_process_flag'
                       ])
    ds = ds.where(Q_mask, np.NAN)
    time = pd.to_datetime(ds.attrs['time_coverage_start'])
    if not hasattr(ds, 'time'):
        ds = ds.assign_coords({'time': time})
        ds = ds.expand_dims(dim='time', axis=0)
    return ds
# endregion

# region Multispectral
def _bands_composite(*bands):
    return np.nanmean(bands, axis=0)

def _unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return np.fliplr((x & mask).astype(bool).astype(int)).reshape(xshape + [num_bits])


@jit(cache=True, nopython=True)
def _rescale(in_array, input_low, input_high, out_low=0, out_high=1):
    return ((in_array - input_low) / (input_high - input_low)) * (out_high - out_low) + out_low


@jit(cache=True, nopython=True)
def _nan_to_zero(array):
    return np.where(np.isnan(array), 0, array)


@jit(cache=True, nopython=True, error_model='numpy')
def _rgb2hsvcpu(R, G, B):
    """
    RGB to HSV
    Convert RGB values to HSV taking into account negative values
    Adapted from https://stackoverflow.com/questions/39118528/rgb-to-hsl-conversion

    :param
    R_ : float
      red channel
    G_ : float
      green channel
    B_ : float
      blue channel
    scale: float, optional

    :returns
    H, S, V : float

    """
    H = np.zeros(R.shape, dtype=np.float32)
    S = np.zeros(R.shape, dtype=np.float32)
    V = np.zeros(R.shape, dtype=np.float32)

    rows, cols = R.shape
    for y in range(0, rows):
        for x in range(0, cols):
            h, s, v = 0, 0, 0

            R_ = R[y, x]
            G_ = G[y, x]
            B_ = B[y, x]

            if np.isnan(R_ * G_ * B_):
                H[y, x] = np.NAN
                S[y, x] = np.NAN
                V[y, x] = np.NAN
                continue

            Cmax = max(R_, G_, B_)
            Cmin = min(R_, G_, B_)
            croma = Cmax - Cmin

            if croma == 0:
                H[y, x] = np.NAN
                S[y, x] = np.NAN
                V[y, x] = np.NAN
                continue

            if Cmax == R_:
                segment = (G_ - B_) / croma
                shift = 0 / 60
                if segment < 0:
                    shift = 360 / 60
                h = segment + shift

            if Cmax == G_:
                segment = (B_ - R_) / croma
                shift = 120 / 60
                h = segment + shift

            if Cmax == B_:
                segment = (R_ - G_) / croma
                shift = 240 / 60
                h = segment + shift

            h *= 60.
            v = Cmax
            s = croma / v

            H[y, x] = h
            S[y, x] = s * 100.
            V[y, x] = v * 100.

    return H, S, V


def _evi(NIR, Red, Blue):
    evi = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    evi.name = 'EVI'
    return evi


def _brightness(Red, Green, Blue):
    return Red + Green + Blue


def _ndvi(NIR, Red):
    da = (NIR - Red) / (NIR + Red)
    da.where((da >= -1) & (da <= 1), np.NAN)
    da.name = 'NDVI'
    return da


def H(array):
    return rgb2hsv(array)


@jit(cache=True, nopython=True)
def _gvi(ndvi, h):
    null_mask = np.logical_or(np.isnan(ndvi), np.isnan(h))

    # this can be substitute with eval(equation)
    vegetated = np.where((h >= (-2354.83 * ndvi) + 522.68), 1, 0)
    semiveg = np.where(
        (h > (-2139.54 * ndvi) + 377.63) & (h < (57.22 * ndvi) + 141.42) & (h < (-2354.83 * ndvi) + 522.68) & (
                h > (-261.64 * ndvi) + 133.30), 1, 0)

    gvi = np.logical_or(vegetated, semiveg)

    gvi_masked = np.where(~null_mask, gvi, np.NaN)

    return gvi_masked


def _decades(data):
    if data[-1] == 1:
        diff = np.diff(data)
        return np.split(data, np.where(np.logical_and(~np.isnan(diff), diff != 0))[0] + 1)[-1].size
    else:
        return -999


def _tif_writer(ds, name, n_date):
    out_path = os.path.join(s.local_folder, 'interm_products', f'ndvi{n_date}')
    ds.name = name
    ds = ds.rio.write_crs("EPSG:4326")
    ds = ds.rename({'lat': 'y', 'lon': 'x'})

    ds_nan = ds.where(~np.isnan(ds), -999).astype(np.int32)
    ds_nan.rio.set_nodata(-999, inplace=True)

    ds_nan.rio.to_raster(out_path, **{'compress': 'lzw',
                                      'interleave': 'band',
                                      'zlevel': 7,
                                      'profile': 'GeoTiff',
                                      'GEOTIFF_VERSION': 1.0,
                                      'bigtiff': 'if_needed'
                                      })


# endregion

@dask.delayed
def filler(date, tile, s):
    print(f'Processing: {date}, {tile}')

    date = pd.to_datetime(date)
    if date.day == 1:
        D10_range = pd.date_range(date,
                                  pd.to_datetime(f'{date.year}{date.month.__str__().zfill(2)}11', format='%Y%m%d'))
    elif date.day == 11:
        D10_range = pd.date_range(date,
                                  pd.to_datetime(f'{date.year}{date.month.__str__().zfill(2)}21', format='%Y%m%d'))
    else:
        if date.month != 12:
            D10_range = pd.date_range(date,
                                      pd.to_datetime(f'{date.year}{str(date.month + 1).zfill(2)}01', format='%Y%m%d'))
        else:
            D10_range = pd.date_range(date, pd.to_datetime(f'{date.year + 1}0101', format='%Y%m%d'))

    tiles_path = []
    for D in D10_range[:-1]:
        yr, mm, str_date = split_date(D)
        absolute_path = os.path.join(s.local_folder, 'archive', yr, mm, str_date, f'*_{tile}_S3*')
        paths = glob.glob(absolute_path)
        for p in paths:
            tiles_path.append(ds_opener(p))

    D10_ds = xr.concat(tiles_path, dim='time', join='override')
    del tiles_path

    meanBlu = xr.apply_ufunc(_bands_composite, D10_ds['Oa03_toc'], D10_ds['Oa04_toc'], join='inner')
    meanGreen = xr.apply_ufunc(_bands_composite, D10_ds['Oa05_toc'], D10_ds['Oa06_toc'], join='inner')
    meanRed = xr.apply_ufunc(_bands_composite, D10_ds['Oa07_toc'], D10_ds['Oa08_toc'], D10_ds['Oa09_toc'],
                             D10_ds['Oa10_toc'], join='inner', )
    meanNIR = xr.apply_ufunc(_bands_composite, D10_ds['Oa16_toc'], D10_ds['Oa17_toc'], D10_ds['Oa18_toc'],
                             join='inner', )
    meanSWIR = xr.apply_ufunc(_bands_composite, D10_ds['S5_an_toc'],
                              join='inner', )

    nominal_coords = {'time': [D10_ds.time[0].values.astype('datetime64[D]')], 'lat': D10_ds.lat, 'lon': D10_ds.lon, }
    nominal_date = nominal_coords['time'][0]
    crs_attrs = D10_ds.crs.attrs
    crs_lat_attrs = D10_ds.crs.lat.attrs
    crs_lon_attrs = D10_ds.crs.lon.attrs

    del D10_ds

    EVI = _evi(meanNIR, meanRed, meanBlu)
    evi_mask = np.isnan(EVI).all(axis=0)
    evi_argmax = EVI.where(~evi_mask, -999).argmax('time', skipna=True).astype(np.uint8)
    max_EVI = EVI.isel({'time': evi_argmax}).assign_coords({'time': nominal_date}).expand_dims(
        dim='time', axis=0).rename('EVI')

    del (EVI, evi_mask, evi_argmax)

    NDVI = _ndvi(meanNIR, meanRed)
    mask = np.isnan(NDVI).all(axis=0)
    argmax = NDVI.where(~mask, -999).argmax('time', skipna=True).astype(np.uint8)

    max_NDVI = NDVI.isel({'time': argmax}).assign_coords({'time': nominal_date}).expand_dims(
        dim='time', axis=0).rename('NDVI').astype(np.float32)
    del(NDVI)
    max_Blu = meanBlu.isel({'time': argmax}).assign_coords({'time': nominal_date}).expand_dims(
        dim='time', axis=0).rename('Blu').astype(np.float32)
    del(meanBlu)
    max_Green = meanGreen.isel({'time': argmax}).assign_coords({'time': nominal_date}).expand_dims(
        dim='time', axis=0).rename('Green').astype(np.float32)
    del(meanGreen)
    max_Red = meanRed.isel({'time': argmax}).assign_coords({'time': nominal_date}).expand_dims(
        dim='time', axis=0).rename('Red').astype(np.float32)
    del(meanRed)
    max_NIR = meanNIR.isel({'time': argmax}).assign_coords({'time': nominal_date}).expand_dims(
        dim='time', axis=0).rename('NIR').astype(np.float32)
    del(meanNIR)
    max_SWIR = meanSWIR.isel({'time': argmax}).assign_coords({'time': nominal_date}).expand_dims(
        dim='time', axis=0).rename('SWIR').astype(np.float32)
    del(meanSWIR, argmax)

    # Blu_rescaled = _rescale(max_Blu.to_numpy(), -0.01, 1.6, 0, 255)
    # Green_rescaled = _rescale(max_Green.to_numpy(), -0.01, 1.6, 0, 255)
    # SWIR_rescaled = _rescale(max_SWIR[0, :, :].to_numpy(), -0.01, 1.6, 0, 1).astype(np.uint8)
    # NIR_rescaled = _rescale(max_NIR[0, :, :].to_numpy(), -0.01, 1.6, 0, 1).astype(np.uint8)
    # RED_rescaled = _rescale(max_Red[0, :, :].to_numpy(), -0.01, 1.6, 0, 1).astype(np.uint8)
    # del (max_Blu, max_Green, max_Red, max_NIR, max_SWIR)

    # h, _, _ = _rgb2hsvcpu(RED_rescaled, NIR_rescaled, SWIR_rescaled)
    # del (SWIR_rescaled, NIR_rescaled, RED_rescaled, _)

    h, _, _ = _rgb2hsvcpu(max_SWIR[0, :, :].to_numpy(), max_NIR[0, :, :].to_numpy(), max_Red[0, :, :].to_numpy())

    h = np.where(mask, np.nan, h).round(2)
    H = xr.DataArray(np.expand_dims(h, 0),
                     coords={'time': nominal_coords['time'],
                             'lon': nominal_coords['lon'].values,
                             'lat': nominal_coords['lat'].values},
                     dims=['time', 'lat', 'lon'],
                     name='H')
    del (h)

    archive = xr.merge([max_Blu, max_Green, max_Red, max_NIR, max_SWIR, H, max_NDVI, max_EVI])
    archive.lat.attrs = crs_lat_attrs
    archive.lon.attrs = crs_lon_attrs
    archive.attrs = crs_attrs

    del(max_EVI, max_Blu, max_Green, max_Red, max_NIR, max_SWIR)

    os.makedirs(os.path.join(s.local_folder, 'interm'), exist_ok=True)
    os.makedirs(os.path.join(s.local_folder, 'interm', nominal_date.__str__().replace('-', '')), exist_ok=True)
    archive_name = os.path.join(os.path.join(s.local_folder,
                                             'interm',
                                             nominal_date.__str__().replace('-', ''),
                                             f'{tile}.nc'))

    archive.to_netcdf(archive_name, format='NETCDF4', compute=True,
                      encoding={'Blu':   {"dtype": "float32", "zlib": True, "complevel": 7},
                                'Green': {"dtype": "float32", "zlib": True, "complevel": 7},
                                'Red':   {"dtype": "float32", "zlib": True, "complevel": 7},
                                'NIR':   {"dtype": "float32", "zlib": True, "complevel": 7},
                                'SWIR':  {"dtype": "float32", "zlib": True, "complevel": 7},
                                'H':     {"dtype": "float32", "zlib": True, "complevel": 7},
                                'NDVI':  {"dtype": "float32", "zlib": True, "complevel": 7},
                                'EVI':   {"dtype": "float32", "zlib": True, "complevel": 7},
                                })

    del (archive)

    # greenness
    gvi = _gvi(max_NDVI[0, :, :].to_numpy(), H[0, :, :].to_numpy())
    del (max_NDVI, H)

    gvi_time = np.expand_dims(gvi, 0)
    gvi_DS = xr.DataArray(gvi_time, coords=nominal_coords, name='GVI').to_dataset()

    zarr_update(gvi_DS, tile, s.archive_path, s.AOI_TL_x, s.AOI_TL_y, s.AOI_BR_x, s.AOI_BR_y)

    return


def zarr_update(gvi, tile, container, AOI_TL_x, AOI_TL_y, AOI_BR_x, AOI_BR_y):
    tile_x = int(tile[1:3])
    tile_y = int(tile[4:6])

    x = tile_x - AOI_TL_x
    y = tile_y - AOI_TL_y

    xi_min = x * 3360  # todo adapt to tile size
    xi_max = xi_min + 3360  # todo adapt to tile size

    yi_min = y * 3360  # todo adapt to tile size
    yi_max = yi_min + 3360  # todo adapt to tile size

    container_DS = xr.open_zarr(container, consolidated=True)

    lat_region = slice(yi_min, yi_max)  # todo pixel is a fix number
    lon_region = slice(xi_min, xi_max)  # todo pixel is a fix number

    with xr.open_zarr(container, consolidated=True) as container_DS:
        pos = np.where(container_DS.time == gvi.time[0])[0].item()
        if pos == container_DS.time.size - 1:
            time_region = slice(pos, None, 1)
        else:
            time_region = slice(pos, pos + 1, 1)

    gvi.to_zarr(container, compute=True, region={'time': time_region, 'lat': lat_region, 'lon': lon_region}, )


if __name__ == '__main__':

    env = platform.system()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-u', '--user', help='User', type=str)
    parser.add_argument('-p', '--password', help='password', type=str)
    parser.add_argument('--hpc', help='switch for hpc', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if env == 'Windows':
        local_folder = r'e:\tmp\S3'
        workers = 1

        x_TL_AOI, y_TL_AOI = 18, 6
        x_BR_AOI, y_BR_AOI = 18, 6

        tiles_exclusion = []
    elif env == 'Linux' and args.hpc is False:
        local_folder = '/wad-3/CGL_Greenness'
        workers = 4

        x_TL_AOI, y_TL_AOI = 16, 4
        x_BR_AOI, y_BR_AOI = 26, 8

        tiles_exclusion = ['X16Y04',
                 'X20Y04', 'X21Y04', 'X22Y04', 'X23Y04', 'X24Y04', 'X25Y04', 'X26Y04',
                 'X24Y07',
                 'X23Y08', 'X24Y08', 'X25Y08', 'X26Y08',
                 'X16Y08', 'X17Y08', 'X18Y08', 'X19Y08', 'X20Y08']

    else:
        local_folder = r'/BGFS/COMMON/maraspi/S3'
        cluster = PBSCluster(cores=4,
                             memory="251gb",
                             resource_spec='select=ncpus=32:mem=251gb',
                             project='DASK_Parabellum',
                             queue='high',
                             local_directory='/local0/maraspi/',
                             walltime='12:00:00',
                             # death_timeout=240,
                             log_directory='/tmp/maraspi/workers/')
        workers = 20

        x_TL_AOI, y_TL_AOI = 16, 4
        x_BR_AOI, y_BR_AOI = 26, 8

        tiles_exclusion = ['X16Y04',
                 'X20Y04', 'X21Y04', 'X22Y04', 'X23Y04', 'X24Y04', 'X25Y04', 'X26Y04',
                 'X24Y07',
                 'X23Y08', 'X24Y08', 'X25Y08', 'X26Y08',
                 'X16Y08', 'X17Y08', 'X18Y08', 'X19Y08', 'X20Y08']

    AOI = [x_TL_AOI, y_TL_AOI, x_BR_AOI, y_BR_AOI]
    pxl_sx = 1 / 336.

    s_date = datetime.datetime.today()
    time_delta = 60

    archive_path = os.path.join(local_folder, 'archive.zarr')
    grid_path = os.path.join(local_folder, 'grid.geojson')
    out_path = os.path.join(local_folder, 'results')
    out_name = 'MVP_S3_300'
    archive_flush = True

    server = 'uservm.vito.be'
    port = 24033
    user = args.user
    password = args.password
    root_path = f'/data/cgl_vol2/SEN3-TOC/'

    s = ProcessSettings(AOI, tiles_exclusion, pxl_sx, grid_path, s_date, time_delta, local_folder, out_path, out_name,
                        archive_path,
                        archive_flush,
                        server, port, user, password, root_path, )
    print('\rPrepare settings')

    # Archive creator/updater
    if os.path.isdir(s.archive_path) and not s.flush:
        s.D10_required = archive_append(s.archive_path, s.D10_range)
    else:
        s.D10_required = archive_creator(s.archive_path, s.D10_range, s.lat_n, s.lon_n, s.minx, s.maxx, s.maxy, s.miny)
    print('\rArchive ready')

    # Data retreat
    if not s.D10_required.empty:
        try:
            asyncio.run(lifter(s))
        except (OSError, asyncssh.Error) as exc:
            sys.exit('SFTP operation failed: ' + str(exc))
    else:
        # TODO adapted and check
        pass

    print('\rLocal data downloaded')

    if env == 'Windows':
        cluster = LocalCluster(n_workers=workers, processes=True, threads_per_worker=1)
        client = Client(cluster)
        client.wait_for_workers(workers)
    elif env == 'Linux' and args.hpc is False:
        cluster = LocalCluster(n_workers=workers, processes=True, threads_per_worker=1,
                               **{'local_directory': '/localdata'})
        client = Client(cluster)
        client.wait_for_workers(workers)
    else:
        cluster.scale(workers)
        client = Client(cluster)
        client.wait_for_workers(7)

    print('\rCluster up and running')

    if not s.D10_required.empty:
        tiles_update = [filler(obs_date, tile, s) for obs_date in s.D10_required.values for tile in s.tile_list]
        dask.compute(tiles_update)
        print('\rArchive updated')

    print('\rcomputing GVI')
    gvi = xr.open_zarr(s.archive_path, consolidated=True).GVI[-6:, :, :]

    gvdm = xr.apply_ufunc(_decades, gvi,
                          input_core_dims=[['time']],
                          exclude_dims={'time', },
                          dask='parallelized',
                          dask_gufunc_kwargs={'allow_rechunk': True},
                          vectorize=True, )

    gvdm.name = 'greenness'
    gvdm = gvdm.rio.write_crs("EPSG:4326")
    gvdm = gvdm.rename({'lat': 'y', 'lon': 'x'})

    gvdm_nan = gvdm.where(~np.isnan(gvdm), -999).astype(np.int32)
    gvdm_nan.rio.set_nodata(-999, inplace=True)

    print('\rWriting output')
    gvdm_nan.rio.to_raster(s.results_path, **{'compress': 'lzw',
                                              'interleave': 'band',
                                              'zlevel': 7,
                                              'profile': 'GeoTiff',
                                              'GEOTIFF_VERSION': 1.0,
                                              'bigtiff': 'if_needed'
                                              })

    print('\rdone')
