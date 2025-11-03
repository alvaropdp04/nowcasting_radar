import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from PIL import Image
import io

def plot_crop_with_map(da_crop, vmin=-10, vmax=80, cmap="turbo"):
    arr = da_crop.values.astype(float)
    arr[arr < -10] = np.nan         
    arr = np.clip(arr, vmin, vmax)

    lats = da_crop.latitude.values
    lons = da_crop.longitude.values

    if np.nanmin(lons) >= 0 and np.nanmax(lons) > 180:
        lons = ((lons + 180) % 360) - 180
        da_crop = da_crop.assign_coords(longitude=lons)

    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    extent = (lon_min, lon_max, lat_min, lat_max)

    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", color="black", linewidth=1)

    im = ax.imshow(
        arr,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),  
        cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.75
    )


    return fig


def crops_to_gif(crops, gif_path, vmin=-10, vmax=80, cmap="turbo", fps=1):
    frames = []
    duration = int(1000 / fps)

    for da_crop in crops:
        fig = plot_crop_with_map(da_crop, vmin=vmin, vmax=vmax, cmap=cmap)  

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight") 
        plt.close(fig)

        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())  
        buf.close()

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )




def radar_to_rgb(array):
    cmap = plt.get_cmap("turbo") 
    norm = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array) + 1e-6)
    rgba = cmap(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)