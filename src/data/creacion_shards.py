import os, io, tarfile
import pandas as pd
import numpy as np
from collections import deque
from data_functions import open_mrms_refd, crop_nyc_300px2


CSV_PATH    = "./data/raw/nombres.csv"     
OUTPUT_DIR  = "./data/raw/shards"           
SHARD_SIZE  = 1000
NPY_COMPRESS = True


os.makedirs(OUTPUT_DIR, exist_ok=True)




    



def write_np_to_tar(tar: tarfile.TarFile, arr: np.ndarray, arcname: str):
    bio = io.BytesIO()
    if NPY_COMPRESS:
        np.savez_compressed(bio, arr=arr)  
        data = bio.getvalue()
        info = tarfile.TarInfo(name=arcname.replace(".npy", ".npz"))
    else:
        np.save(bio, arr)
        data = bio.getvalue()
        info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))

def open_new_shard(idx: int):
    path = os.path.join(OUTPUT_DIR, f"shard-{idx:06d}.tar")
    return tarfile.open(path, mode="w"), path


def build_shards_from_csv():
    shard_idx = 0
    n_in_shard = 0
    global_id  = 0

    tar, path = open_new_shard(shard_idx)
    print(f"[INFO] Abierto shard: {path}")

    buf = deque(maxlen=5)

    df = pd.read_csv(CSV_PATH)
    keys = df["nombres"].tolist()

    for k in keys:
        da = open_mrms_refd(k)          
        da = crop_nyc_300px2(da)
        arr = da.values
        missing = np.isnan(arr)
        arr = np.clip(arr,-10,80)
        arr[missing] = np.nan
        arr = arr.astype(np.float16)
        buf.append(arr)

        if len(buf) < 5:
            continue  

        X = np.stack([buf[0], buf[1], buf[2], buf[3]], axis=0)  # (4,H,W)
        Y = buf[4]                                              # (H,W)

        base = f"{global_id:09d}"
        write_np_to_tar(tar, X, f"{base}.x.npy")
        write_np_to_tar(tar, Y, f"{base}.y.npy")

        n_in_shard += 1
        global_id  += 1

        if n_in_shard >= SHARD_SIZE:
            tar.close()
            print(f"[INFO] Cerrado shard {shard_idx:06d} (samples: {n_in_shard})")
            shard_idx += 1
            tar, path = open_new_shard(shard_idx)
            print(f"[INFO] Abierto shard: {path}")
            n_in_shard = 0


    tar.close()
    print("[OK] Sharding completado.")


build_shards_from_csv()