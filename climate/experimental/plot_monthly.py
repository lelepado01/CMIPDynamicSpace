import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import cartopy.crs as ccrs
import os
import cartopy.feature as cfeature
from easygems.resample import KDTreeResampler
from easygems.show import map_show

NC_FILE   = "error_global_monthly/eof_modes_seasonal.nc" 
OUT_DIR   = NC_FILE.split("/")[0] 
os.makedirs(OUT_DIR, exist_ok=True)
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec",
]
EXTENT    = [-180, 180, -90, 90]       # [lon_min, lon_max, lat_min, lat_max]
CMAP = "RdBu_r"

# ── Load data ─────────────────────────────────────────────────────────────────
ds = nc.Dataset(NC_FILE)
modes_data = ds.variables["modes"][:]   # masked array, shape (12,6,12,64,64)
face_vals  = ds.variables["face"][:]    # (12,)
nlat_vals  = ds.variables["nlat"][:]    # (64,)
nlon_vals  = ds.variables["nlon"][:]    # (64,)
ds.close()

n_months, n_modes, n_face, n_nlat, n_nlon = modes_data.shape
face_idx, nlat_idx, nlon_idx = np.meshgrid(face_vals, nlat_vals, nlon_vals, indexing="ij")  # each shape (12, 64, 64)

face_flat  = face_idx.ravel().astype(int)
nlat_flat  = nlat_idx.ravel().astype(int)
nlon_flat  = nlon_idx.ravel().astype(int)

ref_ds = xr.open_dataset("hpx64_ref_lat_lon.nc")
lon = ref_ds["lon"].values.flatten()  # (12*64*64,)
lat = ref_ds["lat"].values.flatten()  # (12*64*64,)
resampler = KDTreeResampler(lon=lon, lat=lat)

for mode_idx in range(n_modes):
    mode_num = mode_idx + 1

    fig = plt.figure(figsize=(14, 7))
    what = NC_FILE.split("_")[0]
    cond = NC_FILE.split("/")[0].split("_")
    cond = cond[3] if len(cond) > 3 else ""
    fig.suptitle(f"EOF on {what}{cond}, Mode {mode_num}  —  Seasonal Cycle (12 months)", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.01, wspace=0.05)

    # Collect all values for this mode to get a symmetric colour scale
    mode_all = modes_data[:, mode_idx, :, :, :]   # (12, 12, 64, 64)
    vmax = np.nanpercentile(np.abs(mode_all.filled(np.nan)), 98)
    vmin = -vmax

    for month_idx in range(n_months):
        row = month_idx // 4
        col = month_idx  % 4
        ax  = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="k")
        ax.set_extent(EXTENT)

        # Data for this month+mode, flattened to 1-D (face*nlat*nlon,)
        data_2d = modes_data[month_idx, mode_idx, :, :, :]   # (12,64,64)
        data_flat = data_2d.filled(np.nan).ravel()

        map_show(data_flat,ax=ax,cmap=CMAP,resampler=resampler)

        ax.set_title(MONTH_LABELS[month_idx], fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colour bar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=f"EOF Mode {mode_num} amplitude")

    plt.tight_layout()

    out_path = f"{OUT_DIR}/eof_mode_{mode_num:02d}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
