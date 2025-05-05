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
subset(
    dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
    variables=["VHM0", "VMDR", "VTPK","VHM0_SW1","VHM0_WW"],
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_longitude=longitude[0],
    maximum_longitude=longitude[1],
    minimum_latitude=latitude[0],
    maximum_latitude=latitude[1],
    username="aputra7",
    password="Hell4Fire@091",
    output_directory=".",  # Simpan di folder saat ini
    output_filename="wave_data.nc"  # Nama file output
)


# Baca file

ds = xr.open_dataset("wave_data.nc")

# Akses semua variabel
print("Variabel yang tersedia:", list(ds.data_vars))


# 2. Analisis Stokes Drift untuk Identifikasi Pola Arus
def analisis_stokes_drift(dataset):
    # Rumus empiris Stokes drift (Breivik et al., 2016)
    Hs = ds['VHM0']      # Significant wave height (m)
    Tp = ds['VTPK']      # Peak period (s)
    k = 0.16             # Koefisien empiris

    # Stokes drift di permukaan (m/s)
    stokes_u = k * Hs * np.sqrt(9.81 * Hs) * np.cos(np.deg2rad(ds['VMDR']))
    stokes_v = k * Hs * np.sqrt(9.81 * Hs) * np.sin(np.deg2rad(ds['VMDR']))
    
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
              scale=30, color='red')
    plt.title('Stokes Drift Velocity dan Arah')
    plt.show()
    
    return stokes_magnitude, stokes_direction

# 3. Analisis Energi Gelombang
def analisis_energi_gelombang(dataset):
    rho = 1025  # Density of seawater (kg/m³)
    g = 9.81    # Gravitasi (m/s²)
    
    Hs = dataset['VHM0']        # Significant wave height
    Tp = dataset['VTPK']        # Peak period
    
    # Hitung energi gelombang (kW/m)
    energy = (1/64) * rho * g**2 * Hs**2 * Tp / 1000
    
    # Plot
    plt.figure(figsize=(24,16))
    energy.mean(dim=['longitude','latitude']).plot()
    plt.title('Energi Gelombang Temporal')
    plt.ylabel('Energi (kW/m)')
    plt.xlabel('Waktu')
    plt.show()
    
    return energy

# 4. Pemisahan Swell dan Wind Wave
def pemisahan_swell_windwave(dataset):
    # Asumsi variabel tersedia dalam dataset
    swell = dataset['VHM0_SW1']    # Swell wave height
    wind_wave = dataset['VHM0_WW'] # Wind wave height
    
    # Plot perbandingan
    plt.figure(figsize=(24,16))
    swell.mean(dim=['longitude','latitude']).plot(label='Swell')
    wind_wave.mean(dim=['longitude','latitude']).plot(label='Wind Wave')
    plt.title('Perbandingan Swell dan Wind Wave')
    plt.ylabel('Tinggi Gelombang (m)')
    plt.legend()
    plt.show()
    
    return swell, wind_wave

# 5. Prediksi Erosi Pantai
def prediksi_erosi(dataset, stokes_magnitude, energy):
    # Model sederhana berdasarkan CVI (Coastal Vulnerability Index)
    cvi = (energy * stokes_magnitude) / dataset['VHM0']
    
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
    
    return risk_level

# 6. Analisis Longshore Sediment Transport
def analisis_transport_sedimen(dataset):
    theta = np.deg2rad(dataset['VMDR'])  # Arah gelombang dalam radian
    Hs = dataset['VHM0']
    Tp = dataset['VTPK']
    
    # Rumus Kamphuis (1991) untuk transport sedimen
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
    
    return Q

# Eksekusi Analisis
if __name__ == "__main__":
    # Jalankan analisis
    stokes_mag, stokes_dir = analisis_stokes_drift(ds)
    energy = analisis_energi_gelombang(ds)
    swell, wind_wave = pemisahan_swell_windwave(ds)
    risk_level = prediksi_erosi(ds, stokes_mag, energy)
    Q = analisis_transport_sedimen(ds)
    
    
# 7. Animasi untuk Semua Analisis
# Fungsi Estimasi Stokes Drift
def estimasi_stokes_drift(ds):
    Hs = ds['VHM0']
    Tp = ds['VTPK']
    theta = np.deg2rad(ds['VMDR'])
    k = 0.016
    stokes_u = k * Hs * np.sqrt(9.81*Hs) * np.cos(theta)
    stokes_v = k * Hs * np.sqrt(9.81*Hs) * np.sin(theta)
    return stokes_u, stokes_v

# Fungsi Utama Animasi Komprehensif
def animasi_komprehensif(ds, output_file='analisis_gelombang.gif', fps=2, dpi=100, step=4):
    # Hitung semua parameter
    try:
        u, v = ds['VSDX'], ds['VSDY']
    except KeyError:
        u, v = estimasi_stokes_drift(ds)
    
    stokes_mag = np.sqrt(u**2 + v**2)
    Hs = ds['VHM0']
    Tp = ds['VTPK']
    
    # 1. Hitung Energi Gelombang
    rho = 1025  # kg/m³
    g = 9.81    # m/s²
    energy = (1/64) * rho * g**2 * Hs**2 * Tp / 1000  # kW/m
    
    # 2. Hitung Risiko Erosi
    risk = (energy * stokes_mag) / Hs
    
    # 3. Hitung Transport Sedimen
    theta = np.deg2rad(ds['VMDR'])
    Q = 0.00023 * Hs**2.25 * Tp**1.5 * np.sin(2*theta)**0.75
    
    # Setup Figure dan Subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 15),
                        subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle('Analisis Parameter Gelombang Komprehensif', fontsize=16, y=0.92)
    
    # Inisialisasi Plot Awal
    plots = []
    quivers = []
    
    # 1. Stokes Drift (Subplot 0,0)
    norm_stokes = Normalize(vmin=0, vmax=np.nanmax(stokes_mag))
    mesh1 = axs[0,0].pcolormesh(ds.longitude, ds.latitude, stokes_mag[0],
                              cmap='viridis', norm=norm_stokes)
    q1 = axs[0,0].quiver(ds.longitude[::step], ds.latitude[::step],
                        u[0,::step,::step], v[0,::step,::step],
                        scale=5, color='white')
    axs[0,0].set_title('Stokes Drift Velocity (m/s)')
    fig.colorbar(ScalarMappable(norm=norm_stokes, cmap='viridis'),
                ax=axs[0,0], label='Velocity (m/s)')
    
    # 2. Energi Gelombang (Subplot 0,1)
    norm_energy = Normalize(vmin=0, vmax=np.nanmax(energy))
    mesh2 = axs[0,1].pcolormesh(ds.longitude, ds.latitude, energy[0],
                              cmap='plasma', norm=norm_energy)
    axs[0,1].set_title('Energi Gelombang (kW/m)')
    fig.colorbar(ScalarMappable(norm=norm_energy, cmap='plasma'),
                ax=axs[0,1], label='Energi (kW/m)')
    
    # 3. Risiko Erosi (Subplot 1,0)
    norm_risk = Normalize(vmin=0, vmax=np.nanmax(risk))
    mesh3 = axs[1,0].pcolormesh(ds.longitude, ds.latitude, risk[0],
                              cmap='RdYlGn_r', norm=norm_risk)
    axs[1,0].set_title('Indeks Risiko Erosi')
    fig.colorbar(ScalarMappable(norm=norm_risk, cmap='RdYlGn_r'),
                ax=axs[1,0], label='Indeks Risiko')
    
    # 4. Transport Sedimen (Subplot 1,1)
    norm_Q = Normalize(vmin=0, vmax=np.nanmax(Q))
    mesh4 = axs[1,1].pcolormesh(ds.longitude, ds.latitude, Q[0],
                              cmap='jet', norm=norm_Q)
    axs[1,1].set_title('Transport Sedimen (m³/tahun)')
    fig.colorbar(ScalarMappable(norm=norm_Q, cmap='jet'),
                ax=axs[1,1], label='Transport Sedimen')
    
    # Tambahkan fitur peta ke semua subplot
    for ax in axs.flat:
        ax.add_feature(cfeature.LAND, color='lightgrey')
        ax.add_feature(cfeature.OCEAN)
        ax.coastlines(resolution='10m')
        ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                      ds.latitude.min(), ds.latitude.max()],
                      crs=ccrs.PlateCarree())
    
    # Fungsi Update
    def update(frame):
        # Update data
        # 1. Stokes Drift
        mesh1.set_array(stokes_mag[frame].values.ravel())
        q1.set_UVC(u[frame,::step,::step], v[frame,::step,::step])
        
        
        # 2. Energi Gelombang
        mesh2.set_array(energy[frame].values.ravel())
        
        # 3. Risiko Erosi
        mesh3.set_array(risk[frame].values.ravel())
        
        # 4. Transport Sedimen
        mesh4.set_array(Q[frame].values.ravel())
        
        # Update judul waktu
        time_str = str(ds.time[frame].values)[:19]
        fig.suptitle(f'Analisis Gelombang - {time_str}', fontsize=16, y=0.92)
        
        return [mesh1, q1, mesh2, mesh3, mesh4]
    
    # Buat animasi
    ani = FuncAnimation(fig, update, frames=len(ds.time),
                      interval=200, blit=True)
    
    # Simpan sebagai GIF
    print("Menyimpan animasi... (mungkin memerlukan waktu beberapa menit)")
    ani.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    print(f"Animasi disimpan sebagai {output_file}")

if __name__ == "__main__":
    # Baca dataset
    ds = xr.open_dataset("wave_data.nc")
    
    # Jalankan animasi
    animasi_komprehensif(ds, fps=2, step=4)