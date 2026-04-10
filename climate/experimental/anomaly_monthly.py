import xarray as xr
from pathlib import Path

import sys
sys.path.append("climate")
from common import svds_for_anomaly_monthly

MODES = 6
VAR = "t2m0"

targets = xr.open_zarr("hpx64_1983-2017_3h_9varCoupledAtmos-sst.zarr")["targets"]
targets = targets.sel(channel_out=VAR)
targets = targets.sel(time=slice("2000", "2009"))

months = svds_for_anomaly_monthly(targets, modes=MODES)

path_out = Path("./gts_global_monthly_anomalies")
save_monthly_data(months, path_out, modes=MODES)