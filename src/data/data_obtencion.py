import pandas as pd
import time
import calendar
from data_functions import list_mrms_refd_day


inicio = time.time()

keys = []
for year in range(2020,2026):
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
df_guardado.to_csv("./data/raw/nombres.csv")



fin = time.time()

print(f"Tiempo total: {fin - inicio} segundos")

