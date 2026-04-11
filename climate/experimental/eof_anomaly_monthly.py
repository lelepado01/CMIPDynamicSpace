import xarray as xr

import sys
sys.path.append(".")
from climate.utils.pca import svds_for_anomaly_monthly
from climate.utils.plot import save_monthly_data


MODES = 6
VAR = "t2m0"

targets = xr.open_zarr("hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr")["targets"]
targets = targets.sel(channel_out=VAR)
targets = targets.sel(time=slice("2000", "2010"))

months = svds_for_anomaly_monthly(targets, modes=MODES)

save_monthly_data(months, "eof_anomaly_monthly.nc", modes=MODES)