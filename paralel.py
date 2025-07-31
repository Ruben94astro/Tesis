import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para paralelización
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image
import glob
import subprocess
import shutil
import time
from sklearn.preprocessing import normalize
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

# ---- Configuración inicial ----
os.makedirs("frames", exist_ok=True)
os.makedirs("gifs", exist_ok=True)
os.makedirs("light_curve", exist_ok=True)

# [Las funciones fibonacci_sphere, cartesian_from_spherical, ... se mantienen igual]
# ... (Todas tus funciones existentes sin cambios hasta run_simulation)
# ---- Funciones principales ----
def fibonacci_sphere(n_points):
    indices = np.arange(n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return phi, theta % (2*np.pi)

def cartesian_from_spherical(phi, theta, r=1.0):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def spot_mask_geodesic(x, y, z, spot_center, spot_radius_rad):
    center_norm = spot_center / np.linalg.norm(spot_center)
    positions = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(positions, axis=-1, keepdims=True)
    pos_norm = positions / np.clip(norms, 1e-10, None)
    dot_product = np.sum(pos_norm * center_norm, axis=-1)
    dot_product = np.clip(dot_product, -1, 1)
    central_angle = np.arccos(dot_product)
    return np.exp(-(central_angle**2) / (2 * spot_radius_rad**2))

def limbdarkening(u, mu):
    return (1 - u * (1 - mu))

#limb darkening quadratic     
# 
def quadratic(u1, u2, mu):
    return 1 - u1 * (1 - mu) - u2 * (1 - mu)**2


def spot_theta(rotation_period, spot_colatitude, relative_shear):
    latitude = np.pi / 2 - spot_colatitude
    angular_vel_equa = 2*np.pi*u.rad/rotation_period
    return angular_vel_equa*(1-relative_shear*np.sin(latitude)**2)

def add_spots(spots_list, latitude_deg, longitude_deg, radii_deg, rotation_period, relative_shear):
    colatitude_rad = np.deg2rad(90 - latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)
    radii_rad = np.deg2rad(radii_deg)
    ang_vel = spot_theta(rotation_period, colatitude_rad, relative_shear)
    spots_list.append({
        'theta': longitude_rad * u.rad,
        'phi': colatitude_rad,
        'radius': radii_rad,
        'angular_velocity': ang_vel
    })

# def create_gif(input_pattern, output_gif, framerate=15):
#     try:
#         subprocess.run([
#             "ffmpeg", "-y", "-framerate", str(framerate),
#             "-i", input_pattern,
#             "-vf", "scale=800:-1",
#             output_gif
#         ], check=True)
#         return True
#     except (subprocess.CalledProcessError, FileNotFoundError):
#         return False

def gif(input_pattern, output_gif, palette,framerate=17):
    """
    Creating GIF with ffmpeg.
    
    Parameters:
        input_pattern (str): Input pattern of the numbered images (e.g. 'frames/frame_%03d.png')
     output_gif (str): Name of the output GIF file
     palette (str): Name of the temporary palette file
     framerate (int): Frames per second of the GIF
    """
    # color palette
    palette_cmd = ["ffmpeg","-y","-i", input_pattern,"-vf", "palettegen",palette]

    # gif creation
    gif_cmd = ["ffmpeg","-y","-framerate", str(framerate),"-i", input_pattern,"-i", palette,"-lavfi", "paletteuse",output_gif]

    try:
        print("generating color palette")
        subprocess.run(palette_cmd, check=True)

        print("creating gif")
        subprocess.run(gif_cmd, check=True)

        print(f" GIF done: {output_gif}")
    except subprocess.CalledProcessError as e:
        print("something wrong:", e)


def render_frame(i, points, base_intensity, ax, vmin, vmax, spots, cadence_time, r_val, elev, azim):
    ax.clear()
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    intensity = base_intensity.copy()
    
    for spot in spots:
        theta_mov = spot['theta'] + spot['angular_velocity'] * i * cadence_time
        spot_x = r_val * np.sin(spot['phi']) * np.cos(theta_mov.value)
        spot_y = r_val * np.sin(spot['phi']) * np.sin(theta_mov.value)
        spot_z = r_val * np.cos(spot['phi'])
        spot_center = np.array([spot_x, spot_y, spot_z])
        
        mask = spot_mask_geodesic(points[:,0], points[:,1], points[:,2], spot_center, spot['radius'])
        intensity *= (1 -  mask)
    
    ax.scatter(
        points[:,0], points[:,1], points[:,2],
        c=intensity,
        cmap='gray',
        s=1,
        alpha=1,
        vmin=vmin,
        vmax=vmax
    )
    print(f"Procesando frame {i+1}, motion {theta_mov:.2f}",  end='\r')
    max_range = r_val * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    plt.savefig(f"frames/frame_{i:05d}.png", dpi=100, bbox_inches='tight')

def clean_frames():
    for f in glob.glob("frames/*.png"):
        os.remove(f)

def run_simulation(lat, lon, radius, params):
    start_time = time.time()
    print(f"  Simulando: lat={lat}°, lon={lon}°, radius={radius}°")
    
    clean_frames()
    
    spots = []
    add_spots(
        spots, 
        lat, 
        lon, 
        radius,
        params['rotation_period'],
        params['relative_shear']
    )
    
    for i in range(params['total_frames']):
        render_frame(
            i, 
            params['points'], 
            params['base_intensity'], 
            params['ax'], 
            params['vmin'], 
            params['vmax'], 
            spots,
            params['cadence_time'],
            params['r_val'],
            params['elev'],
            params['azim']
        )
    
    gif_name = f"gifs/sim_lat{lat}_lon{lon}_rad{radius}.gif"
    ##if create_gif("frames/frame_%03d.png", gif_name, 15):
    gif("frames/frame_%05d.png", gif_name,"palette.png",17)

    
    frame_files = sorted(glob.glob("frames/*.png"))
    if not frame_files:
        print("    No se encontraron frames para curva de luz")
        return
    
    fluxes = []
    for filename in frame_files:
        try:
            img = Image.open(filename).convert('L')
            img_array = np.array(img, dtype=np.float64)
            fluxes.append(np.sum(img_array))
        except:
            fluxes.append(0)
    

        # Normalized fluxes
    flux_norm = normalize([fluxes], norm="max")[0]
    #flux_norm = np.array(fluxes)/np.max(np.array(fluxes))

    
    df = pd.DataFrame({
        'Frame': range(len(flux_norm)),
        'flux_normalized': flux_norm,
        'Days': np.arange(len(flux_norm)) * params['cadence_time'].to(u.day).value
    })
    
    lc_filename = f"light_curve/lc_lat{lat}_lon{lon}_rad{radius}.csv"
    df.to_csv(lc_filename, index=False)
    print(f"    Curva de luz guardada: {lc_filename}")
    print(f"    Tiempo simulación: {time.time()-start_time:.1f} segundos")

# ---- Parámetros globales de simulación ----
SIMULATION_PARAMS = {
    'r_val': 1.0,
    'n_points': 98000,
    'u1': 0.4,
    'u2': 0.3,
    'rotation_period': 1 * u.day,
    'relative_shear': 0.4,
    'elev': 0,
    'azim': 0,
    'observing_baseline_days': 1 * u.day,
    'cadence_time': 2 * u.minute,
    'vmin':  0.0,
    'vmax':  1.0
}

# ---- Preparación inicial ----
def prepare_simulation():
    print("Preparando componentes de simulación...")
    params = SIMULATION_PARAMS.copy()
    
    phi, theta = fibonacci_sphere(params['n_points'])
    x, y, z = cartesian_from_spherical(phi, theta)
    params['points'] = np.vstack([x, y, z]).T
    
    elev_rad = np.deg2rad(params['elev'])
    azim_rad = np.deg2rad(params['azim'])
    v_x = np.cos(elev_rad) * np.cos(azim_rad)
    v_y = np.cos(elev_rad) * np.sin(azim_rad)
    v_z = np.sin(elev_rad)
    mu = (params['points'][:,0]*v_x + params['points'][:,1]*v_y + params['points'][:,2]*v_z) / params['r_val']
    mu = np.clip(mu, 0, 1)
    
    #params['base_intensity'] = limbdarkening(params['u1'], mu)
    params['base_intensity'] = quadratic(params['u1'], params['u2'], mu)
    
    params['total_frames'] = int((params['observing_baseline_days'] / params['cadence_time']).decompose().value)
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    params['ax'] = fig.add_subplot(111, projection='3d')
    params['ax'].set_box_aspect([1,1,1])
    params['ax'].set_axis_off()
    
    print(f"Configuración lista para {params['total_frames']} frames por simulación")
    return params


# ---- Versión paralela de run_simulation ----
def run_simulation_parallel(args):
    lat, lon, radius, params = args
    # Crear directorio único para esta simulación
    sim_id = f"lat{lat}_lon{lon}_rad{radius}"
    frame_dir = os.path.join("frames", sim_id)
    os.makedirs(frame_dir, exist_ok=True)
    
    start_time = time.time()
    print(f"  Simulando: lat={lat}°, lon={lon}°, radius={radius}°")
    
    spots = []
    add_spots(
        spots, 
        lat, 
        lon, 
        radius,
        params['rotation_period'],
        params['relative_shear']
    )
    
    # Crear figura local para este proceso
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()
    
    for i in range(params['total_frames']):
        # [El resto de la función es igual pero usando frame_dir]
        plt.savefig(os.path.join(frame_dir, f"frame_{i:05d}.png"), dpi=100, bbox_inches='tight')
    
    # [El resto del procesamiento de GIF y curva de luz igual]
    
    # Limpiar frames de esta simulación
    shutil.rmtree(frame_dir)
    
    return lc_filename

# ---- Programa principal paralelizado ----
if __name__ == '__main__':
    sim_params = prepare_simulation()
    
    # Parámetros del grid search
    latitudes = np.arange(-70, 80, 20)
    longitudes = np.arange(0, 360, 60)
    radios = np.arange(0.5, 21, 5)
    
    # Crear lista de tareas
    tasks = []
    for lat in latitudes:
        for lon in longitudes:
            for radius in radios:
                tasks.append((lat, lon, radius, sim_params))
    
    # Ejecutar en paralelo
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_simulation_parallel, task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Completado: {result}")
            except Exception as e:
                print(f"Error en simulación: {str(e)}")
    
    print("\n¡Todas las simulaciones completadas!")
