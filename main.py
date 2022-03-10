import datetime
import asyncio, asyncssh
import sys, os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import dask
import platform
import argparse

from distributed import Client, LocalCluster
from dask_jobqueue import PBSCluster
from urllib.parse import urljoin
from skimage.color import rgb2hsv

np.seterr(all='ignore')


class Settings:

    def __init__(s, AOI, pxl_sz, grid_path, s_date, time_delta,
                 local_folder, out_path, out_name, archive_path, flush,
                 server, port, user, password, root_path):

        # time
        s.start_ref_date = s.__10D_ref_date_finder(s_date)
        s.end_ref_date = s.__10D_ref_date_finder(s.start_ref_date - datetime.timedelta(time_delta))
        s.date_range = s.__date_range(s.start_ref_date, s.end_ref_date)
        s.dek_range = s.__dek_range(s.date_range)
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
        s.tile_list = s.__tile_list(s.x_range, s.y_range)
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
        s.out_name = f'{out_name}_{str(s.dek_range[-2].year)}{str(s.dek_range[-2].month).zfill(2)}{str(s.dek_range[-2].day).zfill(2)}.tif'

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
                grid[f'X{str(xi).zfill(2)}Y{str(yi).zfill(2)}'] = Polygon([(x, y), (x + wide, y), (x + wide, y - length), (x, y - length)])
        return grid

    @classmethod
    def __tile(self, x, y):
        return f'X{str(x).zfill(2)}Y{str(y).zfill(2)}'

    @classmethod
    def __range(self, min, max):
        return range(min, max + 1)

    @classmethod
    def __10D_ref_date_finder(self, date):
        day = date.day
        if 1 <= day <= 11:
            ref_day = 1
        elif 11 <= day <= 21:
            ref_day = 11
        else:
            ref_day = 21
        return datetime.datetime(date.year, date.month, ref_day)

    @classmethod
    def __date_range(self, start, end):
        return [end.date() + datetime.timedelta(days=idate) for idate in range(0, (start - end).days + 1)]

    @classmethod
    def __tile_list(self, x_range, y_range):
        tile_list = []
        for i, v in enumerate(y_range):
            v = [v] * len(x_range)

            tile_list.append(list(map(lambda j: fr'X{str(j[1]).zfill(2)}Y{str(j[0]).zfill(2)}', zip(v, x_range))))

        return sum(tile_list, [])

    @classmethod
    def __dek_range(self, date_range):
        d = [i for i in date_range if i.day in [1, 11, 21]]
        return pd.to_datetime(d)


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


# async def check_local_dek(local_folder, tile_range, dek_range):
#
#     dek_tile_list = []
#
#     for i, dk in enumerate(dek_range):
#         if i == len(dek_range)-1: continue
#
#         path = os.path.join(local_folder, 'tds', dk.strftime('%Y'), dk.strftime('%m'), dk.strftime('%Y%m%d'))
#         for sensor in ['A', 'B']:
#             file_list = glob.glob(os.path.join(path, f'*{sensor}*.nc'))
#
#             if file_list:
#                 tile_list = copy.deepcopy(tile_range)
#                 for f in file_list:
#                     f_tile = os.path.basename(f).split('_')[3]
#                     if f_tile in tile_list:
#                         try:
#                             tile_list.remove(f_tile)
#                         except Exception:
#                             pass
#                 dek_tile_list.append([dk, sensor, tile_list])
#
#             else:
#                 dek_tile_list.append([dk, sensor, tile_range])
#
#     return dek_tile_list


async def download_list(dek_not_available):

    dl_list = []
    for i, dek in enumerate(dek_not_available):
        dk_start, sensor, tile_range = dek
        if i == len(dek_not_available)-1 or sensor == 'B':
            continue

        if dk_start.day == 1:
            dk_end = datetime.datetime.strptime(f'{dk_start.year}{dk_start.month.__str__().zfill(2)}11', '%Y%m%d').date()
        elif dk_start.day == 11:
            dk_end = datetime.datetime.strptime(f'{dk_start.year}{dk_start.month.__str__().zfill(2)}21', '%Y%m%d').date()
        else:
            if dk_start.month == 12:
                dk_end = datetime.datetime.strptime(f'{dk_start.year+1}0101', '%Y%m%d').date()
            else:
                dk_end = datetime.datetime.strptime(f'{dk_start.year}{(dk_start.month+1).__str__().zfill(2)}01', '%Y%m%d').date()

        date_range = [dk_start + datetime.timedelta(days=i) for i in range((dk_end - dk_start).days)]

        for date in date_range:
            dl_list.append([date, tile_range])
            # TODO aD10 local file presence

    return dl_list

# for dt in date_range:
#
#     path = os.path.join(local_folder,'dek' , dt.year, str(dt.month).zfill(2), dt.strftime('%Y%m%d'))
#     if os.path.isdir(path):
#         file_list = glob.glob(os.path.join(path, '*.nc'))
#         if file_list:
#             request_tile = copy.deepcopy(tile_range)
#             filter(lambda f: exclusion(request_tile, f), file_list)
#             download_tile_list.update({path: request_tile})
#     else:
#         continue


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
                        print(f'Worker {task_name} got all files for the {sel_date.year}-{sel_date.month.__str__().zfill(2)}-{sel_date.day.__str__().zfill(2)}')
                        queue.task_done()
                    else:
                        print(f'Worker {task_name} verified that all files for {sel_date.year}-{sel_date.month.__str__().zfill(2)}-{sel_date.day.__str__().zfill(2)} are already in house')
                        queue.task_done()
                except (asyncssh.Error, OSError) as e:
                    raise e


def archive_creator(archive_path, D10_dates, lat_n, lon_n, minx, maxx, maxy, miny):
    time_n = D10_dates[:-1].size

    empty_array = dask.array.empty(shape=(time_n, lat_n, lon_n), chunks=(1, 3360, 3360))
    zero_array = dask.array.zeros_like(empty_array)

    lon = np.round(np.linspace(minx, maxx, lon_n, endpoint=False),8).squeeze()
    lat = np.round(np.linspace(maxy, miny, lat_n, endpoint=False),8).squeeze()

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

        gvi = xr.DataArray(data=zero_array, coords={'time': alignment, 'lat': lat.T, 'lon': lon.T}, name='GVI')

        gvi.to_zarr(archive_path, append_dim='time', consolidated=True)

    return alignment


async def lifter(s):
    # not_avail_dek = await check_local_dek(settings.local_folder, settings.tile_list, settings.dek_range)
    tile2download = await download_list(s.D10_required, s.tile_list)

    tasks = asyncio.Queue(len(tile2download))
    for item in tile2download:
        tasks.put_nowait(item)

    workers = [asyncio.create_task(
        get_files(s.server, s.port, s.user, s.password, s.root_path, s.local_folder, tasks), name=str(i)) for i in
        range(4)]

    await tasks.join()

    for worker in workers:
        worker.cancel()

    results = await asyncio.gather(*workers, return_exceptions=True)

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print('Task %d failed: %s' % (i, str(result)))
        # elif result.exit_status != 0:
        #     print('Task %d exited with status %s:' % (i, result.exit_status))
        #     print(result.stderr, end='')
        else:
            print('Task %d succeeded:' % i)
            # print(result.stdout, end='')


async def download_list(D10_list, tile_range):
    date_range = pd.date_range(D10_list[0], D10_list[-1], freq='D')
    return list(zip(date_range, [tile_range] * date_range.size))


def ds_opener(band_path):
    # ds = xr.open_dataset(band_path, chunks='auto')
    ds = xr.open_dataset(band_path, chunks={'lat': 1680, 'lon': 1680})
    time = pd.to_datetime(ds.attrs['time_coverage_start'])

    if not hasattr(ds, 'time'):
        ds = ds.assign_coords({'time': time})
        ds = ds.expand_dims(dim='time', axis=0)

    return ds


# endregion

# region Multispectral
def _bandscomposit(*args, **kwargs):
    return np.nanmean(args, axis=0)


def _rescale(in_array, input_low, input_high, out_low=0, out_high=1):
    if type(out_low) is int and type(out_high) is int:
        return (((in_array - input_low) / (input_high - input_low)) * (out_high - out_low) + out_low).astype(int)
    else:
        return ((in_array - input_low) / (input_high - input_low)) * (out_high - out_low) + out_low


def _evi(NIR, Red, Blue):
    evi = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    evi.name = 'EVI'
    return evi


def _brightness(Red, Green, Blue):
    return Red + Green + Blue


def _ndvi(NIR, Red):
    da = (NIR - Red) / (NIR + Red)
    da.name = 'NDVI'
    return da


def H(array):
    return rgb2hsv(array)


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

# endregion


@dask.delayed
def filler(date, tile, s):
    print(f'Processing: {date}, {tile}')
    date = pd.to_datetime(date)
    if date.day == 1:
            D10_range = pd.date_range(date, pd.to_datetime(f'{date.year}{date.month}11', format='%Y%m%d'))
    elif date.day == 11:
            D10_range = pd.date_range(date, pd.to_datetime(f'{date.year}{date.month}21', format='%Y%m%d'))
    else:
         if date.month != 12:
            D10_range = pd.date_range(date, pd.to_datetime(f'{date.year}{date.month+1}01', format='%Y%m%d'))
         else:
            D10_range = pd.date_range(date, pd.to_datetime(f'{date.year+1}0101', format='%Y%m%d'))

    tiles_path = []
    for D in D10_range[:-1]:
        yr, mm, str_date = split_date(D)
        absolute_path = os.path.join(s.local_folder, 'archive', yr, mm, str_date, f'*_{tile}_S3A*')
        paths = glob.glob(absolute_path)
        for p in paths:
            tiles_path.append(ds_opener(p))

    D10_ds = xr.concat(tiles_path, dim='time')

    # meanBlu = xr.apply_ufunc(_bandscomposit, D10_ds['Oa03_toc'], D10_ds['Oa04_toc'], join='inner',
    #                          dask='parallelized').rename('mBlue')
    # meanGreen = xr.apply_ufunc(_bandsComposit,D10_ds['Oa05_toc'], D10_ds['Oa06_toc'], join='inner', dask='parallelized').rename('mGreen')
    meanRed = xr.apply_ufunc(_bandscomposit, D10_ds['Oa07_toc'], D10_ds['Oa08_toc'], D10_ds['Oa09_toc'],
                             D10_ds['Oa10_toc'], join='inner', dask='parallelized').rename('mRed').chunk({'time': -1})
    meanNIR = xr.apply_ufunc(_bandscomposit, D10_ds['Oa16_toc'], D10_ds['Oa17_toc'], D10_ds['Oa18_toc'],
                             join='inner', dask='parallelized').rename('mNIR').chunk({'time': -1})
    meanSWIR = xr.apply_ufunc(_bandscomposit, D10_ds['S5_an_toc'],
                              join='inner', dask='parallelized').rename('mSWIR').chunk({'time': -1})

    # EVI = _evi(meanNIR, meanRed, meanBlu)
    # EVI_cleaned = EVI.where(~np.isnan(EVI), -999)

    NDVI = _ndvi(meanNIR, meanRed)
    NDVI_cleaned = NDVI.where(~np.isnan(NDVI), -999)

    argmax = NDVI_cleaned.argmax('time', skipna=True).compute()

    NDVI_cleaned = NDVI.where(NDVI != -999, np.NaN)

    # max_NDVI = NDVI_cleaned.isel({'time': argmax})
    max_NDVI = NDVI_cleaned.max('time', skipna=True)

    # max_Blu = meanBlu.isel({'time': argmax})
    # max_Green = meanGreen.isel({'time': argmax})
    max_Red = meanRed.isel({'time': argmax})
    max_NIR = meanNIR.isel({'time': argmax})
    max_SWIR = meanSWIR.isel({'time': argmax})

    max_Red = max_Red.assign_coords({'time': D10_range[0]}).expand_dims(dim='time', axis=0)
    max_NIR = max_NIR.assign_coords({'time': D10_range[0]}).expand_dims(dim='time', axis=0)
    max_SWIR = max_SWIR.assign_coords({'time': D10_range[0]}).expand_dims(dim='time', axis=0)

    SWIR_rescaled = _rescale(max_SWIR, -0.01, 1.6, 0, 255)
    NIR_rescaled = _rescale(max_NIR, -0.01, 1.6, 0, 255)
    RED_rescaled = _rescale(max_Red, -0.01, 1.6, 0, 255)

    rgb = xr.concat([SWIR_rescaled, NIR_rescaled, RED_rescaled], dim='time')

    rgb = rgb.rename('RGB') \
        .rename({'time': 'band'}) \
        .assign_coords({'band': [1, 2, 3]}) \
        .chunk({'band': 3})

    out = xr.apply_ufunc(rgb2hsv, rgb,
                         input_core_dims=[['band']],
                         output_core_dims=[['HSV']],
                         keep_attrs=False,
                         dask='parallelized',
                         output_dtypes=[float],
                         dask_gufunc_kwargs={'output_sizes': {'HSV': 3}})
    h = out[:, :, 0].rename('H')

    h_squeezed = h.squeeze()

    # h = h_squeezed.assign_coords({'time': D10_range[0]}).expand_dims(dim='time', axis=0)
    # max_NDVI = max_NDVI.assign_coords({'time': D10_range[0]}).expand_dims(dim='time', axis=0)

    # max_Red = max_Red.assign_coords({'time': D10_range[0]})
    # max_Red = max_Red.expand_dims(dim='time', axis=0)
    # max_NIR = max_NIR.assign_coords({'time': D10_range[0]})
    # max_NIR = max_NIR.expand_dims(dim='time', axis=0)
    # max_SWIR = max_SWIR.assign_coords({'time': D10_range[0]})
    # max_SWIR = max_SWIR.expand_dims(dim='time', axis=0)

    # greenness
    gvi = xr.apply_ufunc(_gvi, max_NDVI, h_squeezed,
                         input_core_dims=[['lon', 'lat'], ['lon', 'lat']],
                         output_core_dims=[['lon', 'lat']],
                         dask='parallelized',
                         dask_gufunc_kwargs={'allow_rechunk': True})

    gvi = gvi.assign_coords({'time': D10_range[0]}).expand_dims(dim='time', axis=0).to_dataset(name='GVI')
    gvi = gvi.transpose('time', 'lat', 'lon')
    print(gvi)

    zarr_update(gvi, tile, s.archive_path, s.AOI_TL_x, s.AOI_TL_y, s.AOI_BR_x, s.AOI_BR_y)


def zarr_update(gvi, tile, container, AOI_TL_x, AOI_TL_y, AOI_BR_x, AOI_BR_y):
    tile_x = int(tile[1:3])
    tile_y = int(tile[4:6])

    x = tile_x - AOI_TL_x
    y = tile_y - AOI_TL_y

    xi_min = x * 3360       # todo adapt to tile size
    xi_max = xi_min + 3360  # todo adapt to tile size

    yi_min = y * 3360       # todo adapt to tile size
    yi_max = yi_min + 3360  # todo adapt to tile size

    # t_minx, t_miny, t_maxx, t_maxy = grid[tile].bounds

    container_DS = xr.open_zarr(container, consolidated=True)

    # lat_region = slice(np.where(container_DS.lat == t_maxy)[0].item(), np.where(container_DS.lat == (np.round(t_miny + (1/336)+1, 8)))[0].item()) #todo pixel is a fix number
    # lon_region = slice(np.where(container_DS.lon == t_minx)[0].item(), np.where(container_DS.lon == (np.round(t_maxx - (1/336)+1, 8)))[0].item()) #todo pixel is a fix number

    # if tile_x == AOI_BR_x and tile_y == AOI_BR_y:
    #     lat_region = slice(yi_min, None)  # todo pixel is a fix number
    #     lon_region = slice(xi_min, None)  # todo pixel is a fix number
    # else:
    #     lat_region = slice(yi_min, yi_max + 1)  # todo pixel is a fix number
    #     lon_region = slice(xi_min, xi_max + 1)  # todo pixel is a fix number

    lat_region = slice(yi_min, yi_max)  # todo pixel is a fix number
    lon_region = slice(xi_min, xi_max)  # todo pixel is a fix number

    with xr.open_zarr(container, consolidated=True) as container_DS:
        pos = np.where(container_DS.time == gvi.time[0])[0].item()
        if pos == container_DS.time.size-1:
            time_region = slice(pos, None, 1)
        else:
            time_region = slice(pos, pos+1, 1)

    gvi.to_zarr(container, region={'time': time_region, 'lat': lat_region, 'lon': lon_region})


if __name__ == '__main__':

    env = platform.system()
    HPC = False

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-u', '--user', help='User', type=str)
    parser.add_argument('-p', '--password', help='password', type=str)
    args = parser.parse_args()

    x_TL_AOI, y_TL_AOI = 20, 5
    x_BR_AOI, y_BR_AOI = 21, 6
    AOI = [x_TL_AOI, y_TL_AOI, x_BR_AOI, y_BR_AOI]
    pxl_sx = 1 / 336.

    s_date = datetime.datetime.today()
    # s_date = datetime.datetime.strptime('20220220', '%Y%m%d')
    time_delta = 60

    if env == 'Windows':
        local_folder = r'e:\tmp\S3'
        workers = 1
    elif env == 'Linux' and HPC is False:
        local_folder = '/wad-3/CGL_Greenness'
        workers = 4
    else:
        local_folder = r'/BGFS/COMMON/maraspi/S3'
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

    archive_path = os.path.join(local_folder, 'archive.zarr')
    grid_path = os.path.join(local_folder, 'grid.geojson')
    out_path = os.path.join(local_folder, 'results')
    out_name = 'MVP_S3_300'
    archive_flush = False

    server = 'uservm.vito.be'
    port = 24033
    user = args.user
    password = args.password
    root_path = f'/data/cgl_vol2/SEN3-TOC/'

    s = Settings(AOI, pxl_sx, grid_path, s_date, time_delta, local_folder, out_path, out_name, archive_path, archive_flush,
                 server, port, user, password, root_path, )

    # Archive creator/updater
    if os.path.isdir(s.archive_path) and not s.flush:
        s.D10_required = archive_append(s.archive_path, s.dek_range)
    else:
        s.D10_required = archive_creator(s.archive_path, s.dek_range, s.lat_n, s.lon_n, s.minx, s.maxx, s.maxy, s.miny)

    # Data retreat
    if not s.D10_required.empty:
        try:
            asyncio.run(lifter(s))
        except (OSError, asyncssh.Error) as exc:
            sys.exit('SFTP operation failed: ' + str(exc))
    else:
        #TODO adapted and check
        #[filler(date, tile) for date in s.D10_required for tile in s.tile_list]
        pass

    if env == 'Windows':
        cluster = LocalCluster(n_workers=workers, processes=True)
        client = Client(cluster)
        client.wait_for_workers(workers)
    elif env == 'Linux' and HPC is False:
        cluster = LocalCluster(n_workers=workers, processes=True, threads_per_worker=1, **{'local_directory': '/localdata'})
        client = Client(cluster)
        client.wait_for_workers(workers)
    else:
        cluster.scale(workers)
        client = Client(cluster)
        client.wait_for_workers((workers * 4)/64)

    if not s.D10_required.empty:
        tiles_update = [filler(obs_date, tile, s) for obs_date in s.D10_required.values for tile in s.tile_list]
        dask.compute(tiles_update)

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

    # gvdm = gvdm.rio.reproject("EPSG:4326", nodata=-999, resampling=0)
    gvdm_nan = gvdm.where(~np.isnan(gvdm), -999)
    gvdm_nodata = gvdm_nan.rio.set_nodata(-999)
    gvdm_f = gvdm_nodata.astype(np.int16)

    gvdm.rio.to_raster(s.results_path, **{'compress': 'lzw'})

    print('done')

