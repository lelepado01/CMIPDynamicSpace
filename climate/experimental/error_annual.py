import xarray as xr
import matplotlib.pyplot as plt
from easygems.show import map_show
from pathlib import Path
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os

import sys
sys.path.append("climate")
from common import svds_for_error_yearly, get_hpx_resampler

MODES = 6
VAR = "z500"
OUT_DIR = Path("error_global_alltimes")
os.makedirs(OUT_DIR, exist_ok=True)
EXTENT = [-180, 180, -90, 90]
CMAP = "RdBu_r"
MODE_LABELS = ["Mode " + str(i+1) for i in range(6)]

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

fig = plt.figure(figsize=(12, 10))
fig.suptitle(f"EOF on Error —  Yearly Cycle", fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.1, wspace=0.05)

for mode in range(MODES): 
    row = mode // 2
    col = mode  % 2
    ax  = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="k")
    ax.set_extent(EXTENT)
    
    map_show(v[mode], ax=ax, cmap=CMAP, resampler=get_hpx_resampler())

    ax.set_title(MODE_LABELS[mode], fontsize=10, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])

# Shared colour bar at the bottom
cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=v.min(), vmax=v.max()))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=f"EOF Yearly amplitude")

plt.tight_layout()

out_path = OUT_DIR / "error_yearly.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()