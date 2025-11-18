import s3fs, xarray as xr, numpy as np
from datetime import datetime
import os
import gzip
import shutil
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import time
import glob
import tarfile

fs = s3fs.S3FileSystem(anon=True)


def list_mrms_refd_day(year: int, month: int, day: int,
                       region: str = "CONUS",
                       product: str = "ReflectivityAtLowestAltitude_00.50"):
    ymd = f"{year}{month:02d}{day:02d}"
    prefix = f"noaa-mrms-pds/{region}/{product}/{ymd}/"
    try:
        files = sorted(fs.ls(prefix))
    except FileNotFoundError:
        return []
    return [f for f in files if f.endswith(".grib2") or f.endswith(".grib2.gz")]


def open_mrms_refd(file_s3path: str, *, in_memory: bool = True, keep_temp: bool = False):
    """
    Abre MRMS GRIB2(.gz) desde S3.
    """
    with fs.open(file_s3path, "rb") as src, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".grib2.gz") as tmp_gz:
        shutil.copyfileobj(src, tmp_gz)
        tmp_gz_path = tmp_gz.name

    with gzip.open(tmp_gz_path, "rb") as gz_in, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp_grib:
        shutil.copyfileobj(gz_in, tmp_grib)
        tmp_grib_path = tmp_grib.name

    ds = xr.open_dataset(tmp_grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})

    if not ds.data_vars:
        ds.close()
        for p in (tmp_gz_path, tmp_grib_path):
            if os.path.exists(p): 
                try: 
                    os.remove(p)
                except: 
                    pass
        raise ValueError("Dataset sin variables de datos.")

    varname = list(ds.data_vars)[0]
    da = ds[varname].rename("reflectivity_dbz")

    if "time" not in da.coords and "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})

    if in_memory:
        da = da.load().astype(np.float32)
        ds.close()
        if not keep_temp:
            for p in (tmp_gz_path, tmp_grib_path):
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
        return da


    return da, tmp_gz_path, tmp_grib_path, ds



def crop_nyc_300px2(da: xr.DataArray, size=300,
                   lat_center=40.7128, lon_center=-74.0060):
    """
    Recorta un cuadrado de NxN píxeles centrado sobre Nueva York.
    Funciona con datos MRMS (latitud decreciente), recortando por defecto un tamaño de 300x300 centrado en NY, para facilitar
    el entrenamiento al crear imágenes cuadradas. Recorta el mapa global de 3500x7000 a un área específica (área sobre la que 
    vamos a entrenar el modelo)
    """

    if lon_center < 0:
        lon_center = (lon_center + 360) % 360   # -74 -> 286

    lat_idx = int(np.abs(da.latitude - lat_center).argmin())
    lon_idx = int(np.abs(da.longitude - lon_center).argmin())

    half = size // 2
    da_crop = da.isel(
        latitude=slice(lat_idx - half, lat_idx + half),
        longitude=slice(lon_idx - half, lon_idx + half),
    )

    if da_crop.shape[-2] != size or da_crop.shape[-1] != size:
        raise ValueError(f"El recorte no es {size}x{size}. Resultado: {da_crop.shape}")
    else:
        print("Recorte correcto")
    
    return da_crop




