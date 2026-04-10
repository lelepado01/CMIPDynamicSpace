import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
import xarray as xr

# ── configuration ─────────────────────────────────────────────────────────────

LATLON_REF  = "hpx64_ref_lat_lon.nc"          # NetCDF reference file
TILE_FILES  = [f"{i}.npy" for i in range(12)] # 12 input HPX tiles
OUT_LAT_N   = 180   # output latitude  points  (-90 … 90)
OUT_LON_N   = 360   # output longitude points  (  0 … 360)
OUTPUT_NPY  = "output_latlon.npy"


# ── main conversion ────────────────────────────────────────────────────────────

def convert(latlon_ref=LATLON_REF, tile_files=TILE_FILES, out_lat_n=OUT_LAT_N, out_lon_n=OUT_LON_N, output_npy=OUTPUT_NPY, save=False, plot=False):

    # 1. Load the exact pixel coordinates from the reference file
    ds = nc.Dataset(latlon_ref)
    ref_lat = ds.variables['lat'][:]   # (12, 64, 64)
    ref_lon = ds.variables['lon'][:]   # (12, 64, 64)
    ds.close()

    # 2. Load all 12 data tiles and flatten to scattered points
    if isinstance(tile_files[0], str): 
        faces = [np.load(f) for f in tile_files]
    else: 
        faces = tile_files
    assert all(f.shape == ref_lat.shape[1:] for f in faces), \
        "Tile shape mismatch with reference lat/lon"

    all_lats = ref_lat.ravel()
    all_lons = ref_lon.ravel()
    all_vals = np.stack(faces, axis=0).ravel()

    # print(f"Scattered points : {len(all_lats):,}")
    # print(f"Lat range        : [{all_lats.min():.2f}, {all_lats.max():.2f}]")
    # print(f"Lon range        : [{all_lons.min():.2f}, {all_lons.max():.2f}]")

    # 3. Define output regular grid
    out_lat = np.linspace(-90,  90,  out_lat_n)
    out_lon = np.linspace(  0, 360,  out_lon_n, endpoint=False)
    grid_lon, grid_lat = np.meshgrid(out_lon, out_lat)

    # 4. Bilinear interpolation; fill polar gaps with nearest-neighbour
    points = np.column_stack([all_lons, all_lats])
    result = griddata(points, all_vals, (grid_lon, grid_lat), method='linear')

    nan_mask = np.isnan(result)
    if nan_mask.any():
        # print(f"  Filling {nan_mask.sum()} polar/gap pixels with nearest-neighbour ...")
        fill = griddata(points, all_vals, (grid_lon, grid_lat), method='nearest')
        result[nan_mask] = fill[nan_mask]

    # print(f"Output shape  : {result.shape}")
    # print(f"Value range   : [{result.min():.4f}, {result.max():.4f}]")
    # print(f"NaN remaining : {np.isnan(result).sum()}")

    # 5. Wrap in a DataArray with lat/lon coordinates
    da = xr.DataArray(
        np.expand_dims(result, axis=0),
        dims=["time", "lat", "lon"],
        coords={
            "time": ("time", [1]),
            "lat": ("lat", out_lat, {"units": "degrees_north"}),
            "lon": ("lon", out_lon, {"units": "degrees_east"}),
        },
        name="data",
    )

    # 6. Save
    if save: 
        np.save(output_npy, result)
        print(f"Saved -> {output_npy}")

    # 7. Optional preview
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(result, origin='lower', extent=[0, 360, -90, 90],
                       aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Value')
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_title('HPX64 -> regular lat/lon grid')
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-90, -45, 0, 45, 90])
        plt.tight_layout()
        out_png = output_npy.replace('.npy', '.png')
        plt.savefig(out_png, dpi=150)
        print(f"Plot saved -> {out_png}")
        plt.show()

    return da


# -- entry point ---------------------------------------------------------------

if __name__ == "__main__":
    tile_dir = "."
    files    = [os.path.join(tile_dir, f"{i}.npy") for i in range(12)]
    ref_file = os.path.join(tile_dir, "hpx64_ref_lat_lon.nc")

    da = convert(latlon_ref=ref_file, tile_files=files)
    print(da)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tile_dir = "."
    files    = [os.path.join(tile_dir, f"{i}.npy") for i in range(12)]
    ref_file = os.path.join(tile_dir, "hpx64_ref_lat_lon.nc")

    result = convert(latlon_ref=ref_file, tile_files=files)