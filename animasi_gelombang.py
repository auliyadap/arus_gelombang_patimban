import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from copernicusmarine import subset

# 1. Unduh Data
# Parameter wilayah dan waktu
start_date = "2024-05-01"
end_date = "2024-05-31"
longitude = [106.484, 109.684]  # Batas bujur [barat, timur]
latitude = [-7.1, -4.8]    # Batas lintang [selatan, utara]


# Unduh Dataset Gelombang
#subset(
#    dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
#    variables=["VHM0", "VMDR", "VTPK","VHM0_SW1","VHM0_WW"],
#    start_datetime=start_date,
#    end_datetime=end_date,
#    minimum_longitude=longitude[0],
#    maximum_longitude=longitude[1],
#    minimum_latitude=latitude[0],
#    maximum_latitude=latitude[1],
#    username="aputra7",
#    password="Hell4Fire@091",
#    output_directory=".",  # Simpan di folder saat ini
#    output_filename="wave_data.nc"  # Nama file output
#)


# Baca file

ds = xr.open_dataset("wave_data.nc")

# Akses semua variabel
print("Variabel yang tersedia:", list(ds.data_vars))


# 2. Analisis Stokes Drift untuk Identifikasi Pola Arus
def estimasi_stokes_drift(ds):
    """Estimasi Stokes drift jika tidak ada data langsung"""
    Hs = ds['VHM0']
    Tp = ds['VTPK']
    theta = np.deg2rad(ds['VMDR'])
    
    # Rumus empiris (Liungman & Broström, 2009)
    k = 0.016  # Koefisien empiris
    stokes_u = k * Hs * np.sqrt(9.81 * Hs) * np.cos(theta)
    stokes_v = k * Hs * np.sqrt(9.81 * Hs) * np.sin(theta)
    
    # Hitung magnitude dan arah
    stokes_magnitude = np.sqrt(stokes_u**2 + stokes_v**2)
    stokes_direction = np.rad2deg(np.arctan2(stokes_v, stokes_u)) % 360
    
    # Plot
    plt.figure(figsize=(24,16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    stokes_magnitude[0].plot(cmap='viridis', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.quiver(ds.longitude[::4], ds.latitude[::4],
              stokes_u[0,::4,::4], stokes_v[0,::4,::4],
              scale=7, color='red')
    plt.title('Stokes Drift Velocity dan Arah')
    plt.show()
    
    return stokes_u, stokes_v

def animasi_stokes_drift(ds, output_file='stokes_drift.gif', fps=5, dpi=100):
    """Animasi Stokes Drift Velocity"""
    # Coba ambil data langsung atau estimasi
    try:
        u = ds['VSDX']
        v = ds['VSDY']
    except KeyError:
        u, v = estimasi_stokes_drift(ds)
    
    magnitude = np.sqrt(u**2 + v**2)
    
    # Inisialisasi plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Plot dasar
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(resolution='10m')
    
    # Plot awal
    lon = ds.longitude.values
    lat = ds.latitude.values
    norm = Normalize(vmin=0, vmax=np.nanmax(magnitude))
    mesh = ax.pcolormesh(lon, lat, magnitude[0], cmap='viridis', norm=norm)
    quiver = ax.quiver(lon[::4], lat[::4], u[0,::4,::4], v[0,::4,::4], 
                      scale=5, color='white')
    
    # Colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
    cbar.set_label('Stokes Drift Velocity (m/s)')
    
    def update(frame):
        """Update frame"""
        mesh.set_array(magnitude[frame].values.ravel())
        quiver.set_UVC(u[frame,::4,::4], v[frame,::4,::4])
        ax.set_title(f'Stokes Drift - {ds.time[frame].values}')
        return [mesh, quiver]
    
    ani = FuncAnimation(fig, update, frames=len(ds.time), blit=True)
    ani.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    plt.close()
    print(f"Animasi Stokes drift disimpan sebagai {output_file}")

def animasi_energi_gelombang(ds, output_file='energy_gelombang.gif', fps=5, dpi=100):
    """Animasi Energi Gelombang"""
    # Hitung energi
    rho = 1025  # kg/m³
    g = 9.81    # m/s²
    Hs = ds['VHM0']
    Tp = ds['VTPK']
    energy = (1/64) * rho * g**2 * Hs**2 * Tp / 1000  # kW/m
    
    # Plot
    plt.figure(figsize=(24,16))
    energy.mean(dim=['longitude','latitude']).plot()
    plt.title('Energi Gelombang Temporal')
    plt.ylabel('Energi (kW/m)')
    plt.xlabel('Waktu')
    plt.show()
    
    # Inisialisasi plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(resolution='10m')
    
    # Plot awal
    norm = Normalize(vmin=0, vmax=np.nanmax(energy))
    mesh = ax.pcolormesh(ds.longitude, ds.latitude, energy[0], 
                        cmap='plasma', norm=norm)
    
    # Colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='plasma'), ax=ax)
    cbar.set_label('Energi Gelombang (kW/m)')
    
    def update(frame):
        mesh.set_array(energy[frame].values.ravel())
        ax.set_title(f'Energi Gelombang - {ds.time[frame].values}')
        return [mesh]
    
    ani = FuncAnimation(fig, update, frames=len(ds.time), blit=True)
    ani.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    plt.close()
    print(f"Animasi energi gelombang disimpan sebagai {output_file}")

def animasi_risiko_erosi(ds, output_file='erosi_pantai.gif', fps=5, dpi=100):
    """Animasi Risiko Erosi Pantai"""
    # Hitung komponen risiko
    try:
        stokes_u = ds['VSDX']
        stokes_v = ds['VSDY']
    except KeyError:
        stokes_u, stokes_v = estimasi_stokes_drift(ds)
        
    stokes_mag = np.sqrt(stokes_u**2 + stokes_v**2)
    Hs = ds['VHM0']
    energy = (1/64)*1025*(9.81**2)*Hs**2*ds['VTPK']/1000
    cvi = (energy * stokes_mag) / Hs
    
    # Klasifikasi risiko
    risk_level = xr.where(cvi > 50, 4,  # Sangat Tinggi
                 xr.where(cvi > 30, 3,   # Tinggi
                 xr.where(cvi > 15, 2,   # Sedang
                 1)))                    # Rendah
    
    # Plot peta risiko
    plt.figure(figsize=(24,16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    risk_level[0].plot(cmap='RdYlGn_r', levels=4, 
                      transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.title('Peta Risiko Erosi Pantai')
    plt.show()
    
    # Inisialisasi plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(resolution='10m')
    
    # Plot awal
    norm = Normalize(vmin=0, vmax=np.nanmax(cvi))
    mesh = ax.pcolormesh(ds.longitude, ds.latitude, cvi[0], 
                        cmap='RdYlGn_r', norm=norm)
    
    # Colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='RdYlGn_r'), ax=ax)
    cbar.set_label('Indeks Risiko Erosi')
    
    def update(frame):
        mesh.set_array(cvi[frame].values.ravel())
        ax.set_title(f'Risiko Erosi - {ds.time[frame].values}')
        return [mesh]
    
    ani = FuncAnimation(fig, update, frames=len(ds.time), blit=True)
    ani.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    plt.close()
    print(f"Animasi risiko erosi disimpan sebagai {output_file}")

def animasi_transport_sedimen(ds, output_file='sediment_transport.gif', fps=5, dpi=100):
    """Animasi Transport Sedimen"""
    # Hitung transport sedimen
    theta = np.deg2rad(ds['VMDR'])
    Hs = ds['VHM0']
    Tp = ds['VTPK']
    Q = 0.00023 * Hs**2.25 * Tp**1.5 * np.sin(2*theta)**0.75
    
    # Plot
    plt.figure(figsize=(24,16))
    ax = plt.axes(projection=ccrs.PlateCarree())  # 
    
    Q[0].plot(
        cmap='jet', 
        transform=ccrs.PlateCarree(),  # <-- TRANSFORM WAJIB
        ax=ax  # <-- SPESIFIKASI AXES
    )
    
    ax.coastlines()  # <-- SEKARANG AX SUDAH TERDEFINISI
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.add_feature(cfeature.OCEAN)
    
    plt.title('Longshore Sediment Transport Rate (m³/tahun)')
    plt.show()
    
    # Inisialisasi plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(resolution='10m')
    
    # Plot awal
    norm = Normalize(vmin=0, vmax=np.nanmax(Q))
    mesh = ax.pcolormesh(ds.longitude, ds.latitude, Q[0], 
                        cmap='jet', norm=norm)
    
    # Colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='jet'), ax=ax)
    cbar.set_label('Transport Sedimen (m³/tahun)')
    
    def update(frame):
        mesh.set_array(Q[frame].values.ravel())
        ax.set_title(f'Transport Sedimen - {ds.time[frame].values}')
        return [mesh]
    
    ani = FuncAnimation(fig, update, frames=len(ds.time), blit=True)
    ani.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    plt.close()
    print(f"Animasi transport sedimen disimpan sebagai {output_file}")

if __name__ == "__main__":
    # Baca dataset
    ds = xr.open_dataset("wave_data.nc")
    
    # Buat semua animasi
    animasi_stokes_drift(ds)
    animasi_energi_gelombang(ds)
    animasi_risiko_erosi(ds)
    animasi_transport_sedimen(ds)


