import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from copernicusmarine import subset

# 1. Unduh dan simpan ke file
# Parameter wilayah dan waktu
start_date = "2024-05-01"
end_date = "2024-05-31"
longitude = [106.484, 109.684]  # Batas bujur [barat, timur]
latitude = [-7.1, -4.8]    # Batas lintang [selatan, utara]

#subset(
#    dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
#    variables=["VHM0", "VMDR", "VTPK"],
#    start_datetime=start_date,
#    end_datetime=end_date,
#    minimum_longitude=longitude[0],
#    maximum_longitude=longitude[1],
#    minimum_latitude=latitude[0],
#    maximum_latitude=latitude[1],
#    username="aputra7",
#    password="Hell4Fire@091",
#    output_directory=".",  # Simpan di folder saat ini
#    output_filename="gelombang_data.nc"  # Nama file output
#)


# 2. Baca file NetCDF
ds = xr.open_dataset("wave_data.nc")

# 3. Proses data
hs = ds['VHM0']        # Significant wave height (m)
mwd = ds['VMDR']       # Mean wave direction (derajat)
pp1d = ds['VTPK']      # Peak wave period (s)

# Konversi arah gelombang ke komponen U dan V
radian = np.deg2rad(mwd)
u_wave = -np.sin(radian)  # Tanda negatif untuk arah datang gelombang
v_wave = -np.cos(radian)

# Filter nilai ekstrem
hs_filtered = hs.where(hs < 20)

# 4. Plot Peta
plt.figure(figsize=(24, 16))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([longitude[0], longitude[1], latitude[0], latitude[1]])

# Tambah fitur peta
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN)
ax.coastlines(resolution='10m')

# Plot tinggi gelombang
hs_plot = hs_filtered[0].plot(
    cmap='jet',
    cbar_kwargs={'label': 'Tinggi Gelombang (m)'},
    transform=ccrs.PlateCarree()
)

# Plot arah gelombang
step = 4  # Reduksi kepadatan panah
ax.quiver(
    ds.longitude[::step],
    ds.latitude[::step],
    u_wave[0, ::step, ::step],
    v_wave[0, ::step, ::step],
    scale=30,
    color='white'
)

plt.title(f'Tinggi dan Arah Gelombang\n{start_date} - {end_date}')
plt.show()

# 5. Animasi Temporal
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8))
ax.set_extent([longitude[0], longitude[1], latitude[0], latitude[1]])
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN)
ax.coastlines(resolution='10m')

hs_plot = hs_filtered[0].plot(
    cmap='jet', 
    add_colorbar=False,
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(hs_plot, ax=ax, label='Tinggi Gelombang (m)')

step = 4
quiver = ax.quiver(
    ds.longitude[::step],
    ds.latitude[::step],
    u_wave[0, ::step, ::step],
    v_wave[0, ::step, ::step],
    scale=40,
    color='white'
)

def update(frame):
    hs_plot.set_array(hs_filtered[frame].values.flatten())
    quiver.set_UVC(u_wave[frame, ::step, ::step], v_wave[frame, ::step, ::step])
    ax.set_title(f'Tanggal: {str(ds.time[frame].values)[:19]}')
    return hs_plot, quiver

ani = FuncAnimation(
    fig, 
    update, 
    frames=range(len(hs_filtered.time)), 
    blit=True,
    interval=200
)

ani.save('gelombang_animasi.gif', writer='pillow', fps=5)
print("Animasi disimpan sebagai gelombang_animasi.gif")