import xarray as xr
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import numpy as np
from easygems.resample import HEALPixResampler, KDTreeResampler
from easygems.show import map_show
from pathlib import Path
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import os
import cartopy.feature as cfeature


MODES = 6
VAR = "z500"

PATH = Path("/Users/gabrielepadovani/Downloads/2000-2010")
paths = list(PATH.rglob("*.nc"))

all_times = []
all_gts = []
for path in tqdm(paths): 
    ds = xr.open_dataset(path)
    valid_times = ds.time + ds.step
    ds = ds.assign_coords(valid_time=valid_times)
    ds = ds[VAR].stack(sample=("time", "step")).swap_dims({"sample": "valid_time"})
    all_gts.append(ds - ds.isel(valid_time=0))
    all_times.append(ds)
all_gts = xr.concat(all_gts, dim="valid_time")
all_times = xr.concat(all_times, dim="valid_time")

def svds_on_anomaly(ds): 
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    ds_mon_clim = all_times.groupby("valid_time.month").mean(dim=["valid_time"]) # Calculate the monthly climatology
    ds_anomaly = ds.groupby("valid_time.month") - ds_mon_clim # Calculate the monthly anomalies from the climatology
    ds_anomaly = ds_anomaly.transpose("valid_time", 'face', 'height', 'width')

    ntime, faces, nlat, nlon = ds_anomaly.shape
    data = np.reshape(ds_anomaly.data, (ntime, faces*nlat*nlon))
    u, s, v = svds(data, k=MODES)
    pcs = u*s 
    return pcs, v

ds_anomaly, v = svds_on_anomaly(all_gts)

ref_ds = xr.open_dataset("hpx64_ref_lat_lon.nc")
lon = ref_ds["lon"].values.flatten()  # (12*64*64,)
lat = ref_ds["lat"].values.flatten()  # (12*64*64,)
resampler = KDTreeResampler(lon=lon, lat=lat)

import os
path_out = os.path.basename(__file__)
os.makedirs(path_out, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR   = "./error_monthly_anomalies"                        # directory for saved figures
os.makedirs(OUT_DIR, exist_ok=True)
EXTENT    = [-180, 180, -90, 90]       # [lon_min, lon_max, lat_min, lat_max]
CMAP = "RdBu_r"

MODE_LABELS = ["Mode " + str(i+1) for i in range(6)]
fig = plt.figure(figsize=(12, 10))
fig.suptitle(f"EOF on Error for Anomaly —  Yearly Cycle", fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.1, wspace=0.05)

vmin = v.min()
vmax = v.max()

for mode in range(MODES): 
    row = mode // 2
    col = mode  % 2
    ax  = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="k")
    ax.set_extent(EXTENT)
    
    map_show(
        v[mode],
        ax=ax,
        cmap=CMAP,
        resampler=resampler,
    )

    ax.set_title(MODE_LABELS[mode], fontsize=10, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])

# Shared colour bar at the bottom
cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=f"EOF Yearly amplitude")

plt.tight_layout()

out_path = f"{OUT_DIR}/eof_year.pdf"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

# plt.show()
plt.close()
