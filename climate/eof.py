import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time 
import xeofs as xe

import sys
sys.path.append("./")
from src.hpx2rect import convert
from easygems.resample import HEALPixResampler, KDTreeResampler
from easygems.show import map_show

def orient(tile, k=1, flip=False):
    arr = tile
    arr = np.rot90(arr, k=k)
    if flip:
        arr = np.flipud(arr)
    return arr

def create_eof(path): 
    VAR = "t2m0"
    runs = [path + f for f in os.listdir(path) if not f.endswith(".zip") and f != ".DS_Store"]#[1:2]
    runs = sorted(runs)

    ref_ds = xr.open_dataset("hpx64_ref_lat_lon.nc")
    lon = ref_ds["lon"].values.flatten()  # (12*64*64,)
    lat = ref_ds["lat"].values.flatten()  # (12*64*64,)
    resampler = KDTreeResampler(lon=lon, lat=lat)

    samples = []
    for run in tqdm(runs): 
        ds = xr.open_dataset(f"{run}/artifacts_GR0/forecast_dlwp.nc", engine="netcdf4")

        var = ds[VAR]
        n_steps = var.sizes["step"]

        for step in range(n_steps): 
            time_face = var.isel(step=step).fillna(0.0)
            # time_face = [orient(time_face.isel(face=i).values[0], k=0) for i in range(12)]
            
            hp_resampler = HEALPixResampler(nside=64)
            time_face = time_face.isel(time=0).values.flatten()
            map_show(time_face, resampler=resampler)
            plt.show()
            exit()
            # time_face = convert(ref_file, time_face)
    #         samples.append(time_face)

    # samples = xr.concat(samples, dim="time")
    # model = xe.single.EOF(n_modes=6, standardize=False, use_coslat=False)
    # model.fit(samples, dim="time")

    # for mode in range(6): 
    #     eof = model.components().isel(mode=mode)
    #     eof.plot()
    #     plt.savefig(f"{path.split("_")[-1].replace("/", "")}_{VAR}_{mode}.png", dpi=300)
    #     time.sleep(1)
    #     plt.close()



def main(): 
    create_eof("climate_day/")

    # for path in ["0_zeros/", "1_zeros/", "2_zeros/", "3_zeros/"]: 
    #     print(f"Running {path}")
    #     create_eof(path)

if __name__ == "__main__": 
    main()