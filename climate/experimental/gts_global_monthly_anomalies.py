import xarray as xr
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import numpy as np
from easygems.resample import HEALPixResampler, KDTreeResampler
from easygems.show import map_show
from pathlib import Path
from tqdm import tqdm

MODES = 6
VAR = "z500"

PATH = Path("/Users/gabrielepadovani/Downloads/2000-2010")
paths = list(PATH.rglob("*.nc"))

all_gts = []
all_times = []
for path in tqdm(paths): 
    ds = xr.open_dataset(path)
    valid_times = ds.time + ds.step
    ds = ds.assign_coords(valid_time=valid_times)
    ds = ds[VAR].stack(sample=("time", "step")).swap_dims({"sample": "valid_time"})
    all_gts.append(ds.isel(valid_time=0))
    all_times.append(ds)
all_gts = xr.concat(all_gts, dim="valid_time")
all_times = xr.concat(all_times, dim="valid_time")

def svds_on_anomaly(ds): 
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    ds_mon_clim = all_times.groupby("valid_time.month").mean(dim=["valid_time"]) # Calculate the monthly climatology
    ds_anomaly = ds.groupby("valid_time.month") - ds_mon_clim # Calculate the monthly anomalies from the climatology
    ds_mon_clim = ds_anomaly.transpose("valid_time", 'face', 'height', 'width')
    ds_mon_clim = ds_mon_clim.groupby("valid_time.month")

    months = []
    for m, monthlydata in ds_mon_clim: 
        ntime, faces,  nlat, nlon = monthlydata.shape
        data = np.reshape(monthlydata.data, (ntime, faces*nlat*nlon))
        u, s, v = svds(data, k=MODES)
        pcs = u*s 
        months.append((pcs, v))
    return months

months = svds_on_anomaly(all_gts)

ref_ds = xr.open_dataset("hpx64_ref_lat_lon.nc")
lon = ref_ds["lon"].values.flatten()  # (12*64*64,)
lat = ref_ds["lat"].values.flatten()  # (12*64*64,)
resampler = KDTreeResampler(lon=lon, lat=lat)

import os
path_out = "./gts_global_monthly_anomalies"
os.makedirs(path_out, exist_ok=True)

for i, (ds_anomaly, v) in enumerate(months): 
    for mode in range(MODES): 
        im = map_show(v[mode], resampler=resampler, cmap="RdBu")
        plt.colorbar(im, orientation='horizontal', pad=0.05, fraction=0.05)
        plt.savefig(f"{path_out}/{i}_mode_{mode}.png", dpi=300)
        plt.close()

all_monthly_ds = []
for i, (pcs, v) in enumerate(months):
    month_num = i + 1
    
    # 1. Reshape 'v' back to HEALPix: (mode, face, nlat, nlon)
    v_reshaped = v.reshape(MODES, 12, 64, 64)
    
    # 2. Create a Dataset for this month
    ds_month = xr.Dataset(
        data_vars={
            "modes": (("mode", "face", "nlat", "nlon"), v_reshaped),
            "pcs": (("sample", "mode"), pcs),
        },
        coords={
            "mode": np.arange(MODES),
            "month": month_num,
            "face": np.arange(12),
            "nlat": np.arange(64),
            "nlon": np.arange(64),
        }
    )
    all_monthly_ds.append(ds_month)

# Since 'pcs' have different 'sample' lengths per month, 
# it's often cleanest to save modes separately or save a list of files.
# Let's save the spatial modes (EOFs) which are consistent:

modes_only = [m.drop_vars("pcs") for m in all_monthly_ds]
ds_final = xr.concat(modes_only, dim="month")

ds_final.to_netcdf(f"{path_out}/eof_modes_seasonal.nc")