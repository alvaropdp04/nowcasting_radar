import numpy as np
import io
import glob
import tarfile
import torch 

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})


def generar_freq_pixeles(shards_pattern="./data/raw/shards/*", 
                       bins=np.array([-10,0,10,20,30,40,50,80]), 
                       output_path="./src/utils/freq_pixeles.npy"):
    
    acum = np.zeros(len(bins) - 1)
    files = glob.glob(shards_pattern)

    for i,file in enumerate(files):
        with tarfile.open(file) as tar:
            elementos = tar.getmembers()
            for elemento in elementos:
                f = tar.extractfile(elemento)
                tensor_radar = np.load(io.BytesIO(f.read()))["arr"].flatten()
                counts, _ = np.histogram(tensor_radar, bins)
                acum += counts
        print(f"Archivo {i} procesado")

    np.save(output_path, acum)
    print("Archivo guardado") # SOLO HACE FALTA EJECUTARLO UNA VEZ POR DATASET, YA QUE EL VALOR ES FIJO Y SE GUARDA COMO .npy



# def generar_pesos_perdida(frecuencia_pixeles):
#     """
#     Genera los pesos para la función de pérdida del modelo
#     con el fin de compensar el fuerte desbalance en la distribución
#     de valores de reflectividad (dBZ) en los datos.
#     """
#     pesos_inv = 1/np.array(frecuencia_pixeles)**0.2
#     pesos_norm = pesos_inv/np.mean(pesos_inv)
#     pesos_one = np.ones_like(pesos_norm)
#     pesos_one[0] *= 0.05
#     pesos_one[1] *= 1.25
#     pesos_one[2] *= 1.8
#     pesos_one[3] *= 1.5
#     pesos_one[5] *= 2
#     pesos_one[6] *= 2

#     nuevos_pesos = pesos_norm*pesos_one
#     nuevos_pesos = nuevos_pesos / np.mean(nuevos_pesos)
#     return nuevos_pesos

