import pandas as pd
import calendar
from src.data.data_functions import list_mrms_refd_day
import glob
import webdataset as wds 
import numpy as np
import io
import torch
from torch.utils.data import DataLoader



def generar_nombres_csv(inicio=2020, fin = 2026, ruta_guardado = "./data/raw/nombres.csv"):

    """ Función que genera un archivo csv con los nombres
     determinados de los archivos, para posteriormente generar
     los datos de radar. Recibe como parámetros la fecha de inicio y fin 
     en la que se buscarán los datos de radar en el repositorio"""
    
    keys = []
    for year in range(inicio,fin):
        for month in range(1,13):
            for day in range(1,calendar.monthrange(year,month)[1]+1):
                key = list_mrms_refd_day(year, month, day)
                key_a_añadir = [key[i] for i in range(0, len(key), 10)] # Nos quedamos con datos cada 20 min. Explicado en README
                if key_a_añadir:
                    print(f"Dia {day} del mes {month} del año {year} recuperado")
                    keys.extend(key_a_añadir)
                else:
                    print(f"Dia {day} del mes {month} del año {year} sin datos/no recuperados")
                    

    print(len(keys), "archivos encontrados")

    df_guardado = pd.DataFrame({"nombres": keys})
    df_guardado.index.name = "ID"
    df_guardado.to_csv(ruta_guardado)
    print(f"Archivo guardado en {ruta_guardado}")




def decode_npz(tupla):
    x, y = tupla
    x_decode = torch.from_numpy(np.load(io.BytesIO(x))["arr"])
    x_decode = x_decode.unsqueeze(1)
    y_decode = torch.from_numpy(np.load(io.BytesIO(y))["arr"])
    y_decode = y_decode.unsqueeze(0)
    return x_decode, y_decode


def normalizacion_radar(tupla, min_val = -10, max_val = 80):
    x,y = tupla
    x = torch.nan_to_num(x, nan=0.0)
    y = torch.nan_to_num(y, nan=0.0)
    x_normalizado = (x-min_val) / (max_val-min_val)
    y_normalizado = (y-min_val) / (max_val-min_val)
    x_normalizado = x_normalizado.clip(0.0, 1.0)
    y_normalizado = y_normalizado.clip(0.0, 1.0)
    return x_normalizado, y_normalizado

def generar_dataset(path_shards = "./data/raw/shards/*", props = [0.85,0.1,0.05]):

    """ Función que genera el dataset correspondiente con los datos
    de los shards. Se le pasará como parámetros la ruta donde se encuentran los shards
    y las proporciones en las que se quieran generar los datos de la siguiente manera:

    [prop_entrenamiento, prop_validation, prop_test]

    Al tratarse de datos temporales, se tomarán en orden cronológico los datos, de forma que si
    se determina un 90% de datos de entrenamiento, se tomarán los shards en el intervalo [0, 0.9*(num total de shards)]. Si hay 100 shards,
    se tomarán los 90 primeros para formar el conjunto de entrenamiento. Además, se supone que los datos ya vienen ordenados. Es decir, (shard 0 < shard 1) temporalmente

    IMPORTANTE: al determinar la ruta no se debe únicamente específicar la dirección, si no que
    debe especificarse el tipo de archivo que buscamos (por ejemplo, si queremos todo tipo de archivos en la ruta se debe poner al
    final /*)"""

    if sum(props) != 1:
        raise ValueError("La suma de las proporciones debe ser 1")
    
    archivos = sorted(glob.glob(path_shards))
    num_train = round(props[0]*len(archivos))
    num_val = round(props[1]*len(archivos))
    
    shards_train = archivos[0:num_train]
    shards_val = archivos[num_train: num_train + num_val]
    shards_test = archivos[num_val + num_train : len(archivos)]

    train_dataset = wds.WebDataset(shards_train, shardshuffle = False).shuffle(1000).to_tuple("x.npz", "y.npz").map(decode_npz).map(normalizacion_radar)
    
    val_dataset = wds.WebDataset(shards_val, shardshuffle = False).to_tuple("x.npz", "y.npz").map(decode_npz).map(normalizacion_radar)

    test_dataset = wds.WebDataset(shards_test, shardshuffle = False).to_tuple("x.npz", "y.npz").map(decode_npz).map(normalizacion_radar)

    return train_dataset, val_dataset, test_dataset




def generar_dataloaders(dataset, split = "train", batch_size = 64, num_workers = 8):
    
    """ Función que genera los DataLoaders necesarios para entrenar.
    Recibe como parámetro el dataset, el tipo de loader a crear (train,val,test), el tamaño del batch y el 
    número de procesos paralelos a utilizar """

    if split == "train":
        ds = dataset.batched(batch_size, partial=False)
    elif split in ["val", "test"]:
        ds = dataset.batched(batch_size, partial=True)
    else:
        raise ValueError("El tipo de split debe ser [train, val, test]")

    loader = DataLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=True, persistent_workers= True)

    return loader
