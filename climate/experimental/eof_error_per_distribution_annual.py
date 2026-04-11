import xarray as xr
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np

import sys
sys.path.append(".")
from climate.utils.pca import svds_for_error_yearly
from climate.utils.plot import save_annual_data

VAR = "z500"
PATH = Path("forecasts")
paths = list(PATH.rglob("*.nc"))

targets = xr.open_zarr("hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr")["targets"]
# targets = targets.sel(channel_out=VAR)
targets = targets.sel(time=slice("2000", "2010"))

loss_fn = torch.nn.MSELoss()

per_pixel_error = []
mse_errors = []
for path in tqdm(paths): 
    ds = xr.open_dataset(path)
    valid_times = ds.time + ds.step
    ds = ds.assign_coords(valid_time=valid_times)
    ds = ds[VAR].stack(sample=("time", "step")).swap_dims({"sample": "valid_time"})
    target = targets.sel(time=ds["valid_time"] + pd.Timedelta(hours=6), method="nearest")
    target = target.transpose('face', 'height', 'width', 'valid_time')
    per_pixel_error.append(ds - target)
    mse_errors.append(loss_fn(torch.tensor(ds.to_numpy()), torch.tensor(target.to_numpy())).item())

error_series = pd.Series(mse_errors)
quartiles = error_series.quantile([0.25, 0.5, 0.75])
bins = [-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf]
labels = ['Q1_Low_Error', 'Q2_Mid_Low', 'Q3_Mid_High', 'Q4_High_Error']

for i in range(len(labels)):
    lower_bound = bins[i]
    upper_bound = bins[i+1]
    indices = error_series[(error_series > lower_bound) & (error_series <= upper_bound)].index
    subset_error = [per_pixel_error[i] for i in indices]
    subset_error = xr.concat(subset_error, dim="valid_time")
    
    _, v_q = svds_for_error_yearly(subset_error)
    save_annual_data(v_q, path=f"eof_error_per_distribution_annual_{labels[i]}.pdf", label=f"EOF based on Annual Mean of the Error\n(only for samples in {labels[i]}) for 7-day Forecasts")