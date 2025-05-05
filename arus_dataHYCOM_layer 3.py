# -*- coding: utf-8 -*-
"""
Created on Mon May  5 12:17:29 2025

@author: auliy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  5 12:11:34 2025

@author: auliy
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes

# 1. Membaca Data Arus dari HYCOM
# URL dataset HYCOM
url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

# Membuka dataset tanpa mendekode waktu
ds = xr.open_dataset(url, decode_times=False)

# Periksa atribut variabel waktu 'tau'
print(ds['tau'].attrs)

# kita tentukan tanggal referensi
# Misalnya, asumsi 'analysis' adalah '2023-10-01 00:00:00'
reference_time = np.datetime64('2024-05-01T00:00:00')

# Ubah unit waktu
ds['tau'].attrs['units'] = 'hours since 2024-05-01 00:00:00'

# Dekode waktu
ds = xr.decode_cf(ds)

# Verifikasi variabel waktu 'tau'
print(ds['tau'])

# 2. Menentukan Lokasi dan Rentang Waktu
lat = -6.237055     # Lintang lokasi
lon = 107.920609  # Bujur lokasi
start_time = '2024-05-01T00:00:00'
end_time = '2024-05-31T23:59:59'

# Membuat rentang waktu setiap 6 jam
times = pd.date_range(start=start_time, end=end_time, freq='6H')

# 3. Mengetahui Kedalaman Permukaan
surface_depth = ds['depth'][2]  # Memilih kedalaman pertama (permukaan)

# 4. Mengambil Data Arus dan Menyimpan ke List
data_list = []

for time in times:
    # Memilih data arus pada waktu, kedalaman, dan lokasi tertentu
    data_u = ds['water_u'].sel(time=time, depth=surface_depth, lat=lat, lon=lon, method='nearest').values
    data_v = ds['water_v'].sel(time=time, depth=surface_depth, lat=lat, lon=lon, method='nearest').values

    # Memastikan data_u dan data_v adalah skalar
    if data_u.size == 1 and data_v.size == 1:
        u_value = data_u.item()
        v_value = data_v.item()

        # Menghitung arah arus
        direction = (np.degrees(np.arctan2(u_value, v_value)) + 360) % 360

        # Menyimpan data ke list
        data_list.append({
            'datetime': time,
            'latitude': lat,
            'longitude': lon,
            'u': u_value,
            'v': v_value,
            'direction': direction
        })
    else:
        print(f"Data pada waktu {time} tidak tersedia atau memiliki ukuran yang tidak sesuai.")

# 5. Membuat DataFrame dari List Data
data = pd.DataFrame(data_list)

# 6. Membuat Kolom Waktu Terpisah
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['second'] = data['datetime'].dt.second

# 7. Menyusun DataFrame untuk Disimpan
data_to_save = data[['year', 'month', 'day', 'hour', 'minute', 'second',
                     'latitude', 'longitude', 'u', 'v', 'direction']]

data['speed'] = np.sqrt(data['u']**2 + data['v']**2)

data['direction_calculated'] = (np.degrees(np.arctan2(data['u'], data['v'])) + 360) % 360

data['direction'] = data['direction_calculated']
data.drop(columns=['direction_calculated'], inplace=True)


# 8. Menyimpan Data ke File .txt
data_to_save.to_csv('data_arus.txt', sep=' ', index=False, header=False, float_format='%.4f')

print("Data berhasil disimpan ke 'data_arus_3.txt'")


# Assuming your data is loaded into a DataFrame named 'data'
# and the datetime is combined into a single 'datetime' column

plt.figure(figsize=(15, 5))
plt.quiver(
    data['datetime'],
    np.zeros_like(data['u']),  # All vectors start from zero on the y-axis
    data['u'],
    data['v'],
    angles='uv',
    scale_units='xy',
    scale=1,
    width=0.003,
    headwidth=3,
    headlength=5,
    headaxislength=4.5
)

plt.title('Current Vectors Over Time')
plt.xlabel('Time')
plt.ylabel('Current Vector')
plt.yticks([])
plt.grid(True)
plt.show()

# Membuat rose diagram
plt.figure(figsize=(8, 8))
ax = WindroseAxes.from_ax()
ax.bar(data['direction'], data['speed'], normed=True, opening=0.8, edgecolor='white')
ax.set_title('Rose Diagram Arus')
ax.set_legend()
plt.show()
