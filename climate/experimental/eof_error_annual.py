import xarray as xr
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(".")
from climate.utils.pca import svds_for_error_yearly
from climate.utils.plot import save_annual_data

VAR = "t2m0"

PATH = Path("2000-2010")
paths = list(PATH.rglob("*.nc"))

targets = xr.open_zarr("hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr")["targets"]
targets = targets.sel(channel_out=VAR)
targets = targets.sel(time=slice("2000", "2010"))

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

ds_anomaly, v = svds_for_error_yearly(all_gts)

save_annual_data(v, path="eof_error_annual.pdf", label="EOF based on Annual Mean of the Error for 7-day Forecasts")