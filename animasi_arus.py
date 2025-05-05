import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation

# Membaca Data Arus dari HYCOM
# URL dataset HYCOM
url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

# Membuka dataset tanpa mendekode waktu
ds = xr.open_dataset(url, decode_times=False)

# Periksa atribut variabel waktu 'tau'
print(ds['tau'].attrs)

# Misalkan kita tentukan tanggal referensi
reference_time = np.datetime64('2024-05-06T00:00:00')

# Ubah unit waktu
ds['tau'].attrs['units'] = 'hours since 2024-05-06 00:00:00'

# Dekode waktu
ds = xr.decode_cf(ds)

# Verifikasi variabel waktu 'tau'
print(ds['tau'])  


# Subset wilayah Indonesia
lon_min, lon_max = 106.6, 108.2   # Longitude Indonesia
lat_min, lat_max = -6.3, -5.3     # Latitude Indonesia

ds_subset = ds.sel(
    lon=slice(lon_min, lon_max),
    lat=slice(lat_min, lat_max),
    time=slice("2024-05-01", "2024-05-31")  # Pilih rentang waktu 1 bulan
)

# Ambil hanya data permukaan (depth=0) untuk menghemat memori
if "depth" in ds_subset.dims:
    ds_subset = ds_subset.sel(depth=0)

# Pilih hanya variabel yang dibutuhkan (arus zonal "u" dan meridional "v")
ds_subset = ds_subset[["water_u", "water_v"]]

# Konversi data ke float32 untuk mengurangi ukuran memori
ds_subset["water_u"] = ds_subset["water_u"].astype(np.float32)
ds_subset["water_v"] = ds_subset["water_v"].astype(np.float32)

# Simpan hasilnya dalam format yang lebih kecil
output_file = "hycom_subset_mei2024.nc"
ds_subset.to_netcdf(output_file, format="NETCDF4")

print(f"✅ Data berhasil diproses dan disimpan sebagai {output_file}")

# Load dataset HYCOM
ds = xr.open_dataset('hycom_subset_mei2024.nc')  

# Ambil data u, v, lon, lat untuk periode 1-31 Januari 2024
u_jan = ds['water_u'].sel(time=slice("2024-05-01", "2024-05-31"))
v_jan = ds['water_v'].sel(time=slice("2024-05-01", "2024-05-31"))
lon = ds['lon']
lat = ds['lat']
times = ds['time'].sel(time=slice("2024-05-01", "2024-05-31"))

# Pastikan masking daratan (opsional)
# Buka dataset GEBCO
gebco = xr.open_dataset('gebco_2024_n-5.3964_s-6.3873_w106.6098_e108.2271.nc')  

batimetri = gebco['elevation'].interp(lat=ds['lat'], lon=ds['lon'])

# Masking berdasarkan batimetri (hapus jika elevasi >= 0)
u_masked = u_jan.where(batimetri < 0)
v_masked = v_jan.where(batimetri < 0)

# Setup figure
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
ax.set_extent([106.6, 108.2, -6.3, -5.3], crs=ccrs.PlateCarree())  # Wilayah Indonesia
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.COASTLINE)

# Filter hanya kedalaman negatif (laut)
depth_masked = np.where(batimetri < 0, batimetri, np.nan)

# Plot kontur kedalaman dengan colormap yang sesuai
contour = ax.contourf(lon, lat, depth_masked, cmap='Blues_r', levels=50, alpha=0.7)
plt.colorbar(contour, ax=ax, label="Kedalaman (m)")


# Inisialisasi quiver plot
q = ax.quiver(lon, lat, u_masked.isel(time=0), v_masked.isel(time=0), cmap="coolwarm", scale=2)

# Teks untuk menampilkan tanggal
title = ax.set_title(f"Pergerakan Arus Laut - {times[0].dt.strftime('%Y-%m-%d').values}")

# Fungsi update animasi
def update(frame):
    global q
    q.remove()  # Hapus quiver sebelumnya
    q = ax.quiver(lon, lat, u_masked.isel(time=frame), v_masked.isel(time=frame),cmap="coolwarm", scale=10)
    title.set_text(f"Pergerakan Arus Laut - {times[frame].dt.strftime('%Y-%m-%d').values}")
    
    
# Buat animasi
ani = FuncAnimation(fig, update, frames=len(times), interval=200)

# Simpan sebagai video atau GIF (opsional)
ani.save("animasi_arus.gif", writer="pillow", fps=5)
ani.save("animasi_arus.mp4", writer='ffmpeg', fps=5)

plt.show() 
# %%