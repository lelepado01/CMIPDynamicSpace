from scipy.sparse.linalg import svds
import numpy as np
import xarray as xr
from easygems.resample import KDTreeResampler

import sys
sys.path.append(".")
from climate.consts import *

def get_hpx_resampler(): 
    ref_ds = xr.open_dataset("hpx64_ref_lat_lon.nc")
    lon = ref_ds["lon"].values.flatten()  # (12*64*64,)
    lat = ref_ds["lat"].values.flatten()  # (12*64*64,)
    return KDTreeResampler(lon=lon, lat=lat)

def svds_for_error_yearly(ds, modes=6): 
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    ds = ds.transpose("valid_time", 'face', 'height', 'width')

    ntime, faces, nlat, nlon = ds.shape
    data = np.reshape(ds.data, (ntime, faces*nlat*nlon))
    if hasattr(data, "compute"): 
        data = data.compute()
    u, s, v = svds(data, k=modes)
    pcs = u*s 
    return pcs, v

def svds_for_error_monthly(ds, modes=6): 
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    ds = ds.transpose("valid_time", 'face', 'height', 'width')
    ds = ds.groupby("valid_time.month")

    months = []
    for m, monthlydata in ds: 
        ntime, faces,  nlat, nlon = monthlydata.shape
        data = np.reshape(monthlydata.data, (ntime, faces*nlat*nlon))
        if hasattr(data, "compute"): 
            data = data.compute()
        u, s, v = svds(data, k=modes)
        pcs = u*s 
        months.append((pcs, v))
    return months

def svds_for_anomaly_yearly(ds, modes=6): 
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    ds_mon_clim = ds.groupby("time.year").mean(dim=["time"]) # Calculate the monthly climatology
    ds_anomaly = ds.groupby("time.year") - ds_mon_clim # Calculate the monthly anomalies from the climatology
    ds_anomaly = ds_anomaly.transpose("time", 'face', 'height', 'width')

    ntime, faces, nlat, nlon = ds_anomaly.shape
    data = np.reshape(ds_anomaly.data, (ntime, faces*nlat*nlon))
    if hasattr(data, "compute"): 
        data = data.compute()
    u, s, v = svds(data, k=modes)
    pcs = u*s 
    return pcs, v

def svds_for_anomaly_monthly(ds, modes=6): 
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    ds_mon_clim = ds.groupby("time.month").mean(dim=["time"]) # Calculate the monthly climatology
    ds_anomaly = ds.groupby("time.month") - ds_mon_clim # Calculate the monthly anomalies from the climatology
    ds_mon_clim = ds_anomaly.transpose("time", 'face', 'height', 'width')
    ds_mon_clim = ds_mon_clim.groupby("time.month")

    months = []
    for m, monthlydata in ds_mon_clim: 
        ntime, faces,  nlat, nlon = monthlydata.shape
        data = np.reshape(monthlydata.data, (ntime, faces*nlat*nlon))
        if min(data.shape) < modes: continue
        if hasattr(data, "compute"): 
            data = data.compute()
        u, s, v = svds(data.compute(), k=modes)
        pcs = u*s 
        months.append((pcs, v))
    return months