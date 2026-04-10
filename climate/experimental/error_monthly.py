import xarray as xr
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append("climate")
from common import svds_for_error_monthly, save_monthly_data

MODES = 6
VAR = "t2m0"
PATH = Path("2000-2010")
paths = list(PATH.rglob("*.nc"))

targets = xr.open_zarr("hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr")["targets"]
targets = targets.sel(channel_out=VAR)
targets = targets.sel(time=slice("2000", "2009"))

all_gts = []
for path in tqdm(paths): 
    ds = xr.open_dataset(path)
    valid_times = ds.time + ds.step
    ds = ds.assign_coords(valid_time=valid_times)
    ds = ds[VAR].stack(sample=("time", "step")).swap_dims({"sample": "valid_time"})
    target = targets.sel(time=ds["valid_time"] + pd.Timedelta(hours=6), method="nearest")
    target = target.transpose('face', 'height', 'width', 'valid_time')
    all_gts.append(ds - target)
all_gts = xr.concat(all_gts, dim="valid_time")

months = svds_for_error_monthly(all_gts)

path_out = "./error_global_monthly"
save_monthly_data(months, path_out, modes=MODES)