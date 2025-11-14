from .data_functions import open_mrms_refd, crop_nyc_300px2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from PIL import Image
from ..utils.draw_functions import crops_to_gif, wrap_with_coords_2d
import io, tarfile
import xarray as xr

archivos = pd.read_csv("./data/raw/nombres.csv")






