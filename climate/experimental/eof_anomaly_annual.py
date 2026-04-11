import xarray as xr

import sys
sys.path.append(".")
from climate.utils.pca import svds_for_anomaly_yearly
from climate.utils.plot import save_annual_data

MODES = 6
VAR = "t2m0"

targets = xr.open_zarr("hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr")["targets"]
targets = targets.sel(channel_out=VAR)
targets = targets.sel(time=slice("2000", "2010"))

ds_anomaly, v = svds_for_anomaly_yearly(targets, modes=MODES)

save_annual_data(v, path="eof_anomaly_annual.pdf", label="EOF based on Annual Mean of the GT for 7-day Forecasts")
