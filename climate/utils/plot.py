import numpy as np
import os
import xarray as xr
from easygems.show import map_show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.append(".")
from climate.utils.pca import get_hpx_resampler
from climate.consts import *

def save_annual_data(data, path="viz.pdf", modes=6, label=""): 
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(label, fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.1, wspace=0.05)

    for mode in range(modes): 
        row = mode // 2
        col = mode  % 2
        ax  = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="k")
        ax.set_extent(EXTENT)
        
        map_show(data[mode], ax=ax, cmap=CMAP, resampler=get_hpx_resampler())

        ax.set_title(MODE_LABELS[mode], fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=data.min(), vmax=data.max()))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="EOF Yearly amplitude")

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_monthly_data(months, path="data.nc", modes=6): 
    os.makedirs(path, exist_ok=True)

    all_monthly_ds = []
    for i, (pcs, v) in enumerate(months):
        month_num = i + 1
        v_reshaped = v.reshape(modes, 12, 64, 64)
        ds_month = xr.Dataset(
            data_vars={
                "modes": (("mode", "face", "nlat", "nlon"), v_reshaped),
                "pcs": (("sample", "mode"), pcs),
            },
            coords={
                "mode": np.arange(modes),
                "month": month_num,
                "face": np.arange(12),
                "nlat": np.arange(64),
                "nlon": np.arange(64),
            }
        )
        all_monthly_ds.append(ds_month)

    modes_only = [m.drop_vars("pcs") for m in all_monthly_ds]
    ds_final = xr.concat(modes_only, dim="month")
    ds_final.to_netcdf(path)
