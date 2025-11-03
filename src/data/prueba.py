from .data_functions import open_mrms_refd, crop_nyc_300px2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from PIL import Image
from ..utils.draw_functions import crops_to_gif

archivos = pd.read_csv("./data/raw/nombres.csv")

# for archivo in archivos["nombres"]:
#     da = open_mrms_refd(archivo)
#     da_crop = crop_nyc_300px2(da = da)
#     da_value = da_crop.values


#     print(da_value.min(), da_value.max())
#     print(da_value)


archivo5 = archivos["nombres"][150]
archivo6 = archivos["nombres"][151]
archivo7 = archivos["nombres"][152]
archivo8 = archivos["nombres"][153]
archivo9 = archivos["nombres"][154]
archivo10 = archivos["nombres"][155]
archivo11 = archivos["nombres"][156]
archivo12 = archivos["nombres"][157]

da5 = open_mrms_refd(archivo5)
da_crop5 = crop_nyc_300px2(da = da5)
da_value5 = da_crop5.values
da_value5[da_value5 < -10.0] = np.nan
da_value5 = np.clip(da_value5, -10, 80)

da6 = open_mrms_refd(archivo6)
da_crop6 = crop_nyc_300px2(da = da6)
da_value6 = da_crop6.values
da_value6[da_value6 < -10.0] = np.nan
da_value6 = np.clip(da_value6, -10, 80)

da7 = open_mrms_refd(archivo7)
da_crop7 = crop_nyc_300px2(da = da7)
da_value7 = da_crop7.values
da_value7[da_value7 < -10.0] = np.nan
da_value7 = np.clip(da_value7, -10, 80)

da8 = open_mrms_refd(archivo8)
da_crop8 = crop_nyc_300px2(da = da8)
da_value8 = da_crop8.values
da_value8[da_value8 < -10.0] = np.nan
da_value8 = np.clip(da_value8, -10, 80)

da9 = open_mrms_refd(archivo9)
da_crop9 = crop_nyc_300px2(da = da9)
da_value9 = da_crop9.values
da_value9[da_value9 < -10.0] = np.nan
da_value9 = np.clip(da_value9, -10, 80)

da10 = open_mrms_refd(archivo10)
da_crop10 = crop_nyc_300px2(da = da10)
da_value10 = da_crop10.values
da_value10[da_value10 < -10.0] = np.nan
da_value10 = np.clip(da_value10, -10, 80)

da11 = open_mrms_refd(archivo11)
da_crop11 = crop_nyc_300px2(da = da11)
da_value11 = da_crop11.values
da_value11[da_value11 < -10.0] = np.nan
da_value11 = np.clip(da_value11, -10, 80)

da12 = open_mrms_refd(archivo12)
da_crop12 = crop_nyc_300px2(da = da12)
da_value12 = da_crop12.values
da_value12[da_value12 < -10.0] = np.nan
da_value12 = np.clip(da_value12, -10, 80)





imagenes = [da_crop5, da_crop6, da_crop7, da_crop8, da_crop9, da_crop10, da_crop11, da_crop12]

# frames = []
# for arr in imagenes:
#     frames.append(Image.fromarray(radar_to_rgb(arr)))

# frames[0].save("animacion.gif", append_images = frames[1:], save_all = True, loop = 0, duration = 1000)



crops_to_gif(imagenes, "animacion.gif")