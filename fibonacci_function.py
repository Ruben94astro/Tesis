
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
import os
from PIL import Image
import glob
import subprocess
import matplotlib as mpl

os.makedirs("frames", exist_ok=True)

# ---- fibonacci sphere ----
def fibonacci_sphere(n_points):
    """generating equal points to distributing in sphere"""
    indices = np.arange(n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_points)  # [0, π]
    theta = np.pi * (1 + 5**0.5) * indices  # aureo angle
    
    return phi, theta % (2*np.pi)  # etting values between 0 and 2π

def cartesian_from_spherical(phi, theta, r=1.0):
    """Convert spherical coordinates to cartesians"""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def spot_mask_geodesic(x, y, z, spot_center, spot_radius_rad):
    """Máscara usando distancia geodésica real con optimización"""
    # Convert the spot_center vector to a unit vector (of length 1).
    center_norm = spot_center / np.linalg.norm(spot_center)
    
    # Product
    positions = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(positions, axis=-1, keepdims=True)
    pos_norm = positions / np.clip(norms, 1e-10, None)  #np.clip avoids division by zero (if a vector had length 0).
    
    dot_product = np.sum(pos_norm * center_norm, axis=-1) #Calculation of the Product Point
    dot_product = np.clip(dot_product, -1, 1) #Force the values to be between [-1, 1]. This is necessary because numerical errors may produce values slightly outside this range.
    
    # central_angle is the geodesic distance (in radians) between each point on the surface and the center of the spot.
    central_angle = np.arccos(dot_product)
    
    return np.exp(-(central_angle**2) / (2 * spot_radius_rad**2))

#function to calculate angular velocity  
def spot_theta(rotation_period, spot_colatitude, relative_shear):
    ''' Parameters:
    - rotation_period: rotation period of the sta
    - spot_colatitude: latitud of spot(radians)
    - relative_shear: parameter between pole and equator
    Returns:
    - angular velocity. '''
    
    latitude = np.pi / 2 - spot_colatitude
    angular_vel_equa = 2*np.pi*u.rad/rotation_period
    angular_velocity = angular_vel_equa*(1-relative_shear*np.sin(latitude)**2)
    return angular_velocity

def limbdarkening(u, mu):
    ''' lineal limb darkening'''
    return (1 - u * (1 - mu))

def add_spots(latitude_deg, longitude_deg, radii_deg):
    colatitude_rad = np.deg2rad(90 - latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)
    radii_rad = np.deg2rad(radii_deg)
    ang_vel = spot_theta(rotation_period, colatitude_rad, 0.2)  
    spots.append({
        'theta': longitude_rad * u.rad,
        'phi': colatitude_rad,
        'radius': radii_rad,
        'angular_velocity': ang_vel
    })

def gif(input_pattern="frames/frame_%03d.png", output_gif="output.gif", 
        palette="palette.png", framerate=17):
    palette_cmd = [
        "ffmpeg", "-y", "-i", input_pattern,
        "-vf", "palettegen", palette
    ]
    gif_cmd = [
        "ffmpeg", "-y", "-framerate", str(framerate),
        "-i", input_pattern, "-i", palette,
        "-lavfi", "paletteuse", output_gif
    ]
    try:
        subprocess.run(palette_cmd, check=True)
        subprocess.run(gif_cmd, check=True)
        print(f"GIF creado: {output_gif}")
    except subprocess.CalledProcessError as e:
        print("Error en ffmpeg:", e)

def flux_plot():
    '''
    Function take a list normalizing the flux, converting the list in a csv file and rename the columns
    and return a plot 

    '''
    frame_files = sorted(glob.glob("frames/frame_*.png"))
    fluxes = []

    for filename in frame_files:
        img = Image.open(filename).convert('L')  # Grayscale
        img_array = np.array(img, dtype=np.float64)
        flux_total = np.sum(img_array)
        fluxes.append(flux_total)

    # Normalized fluxes
    flux_norm = normalize([fluxes], norm="max")[0]
    df = pd.DataFrame(flux_norm)

    # Creating columns
    df.index.name = 'Frame'
    df.reset_index(inplace=True)
    #changing frames for days
    df['Days'] = df['Frame'] *(cadence_time.to(u.day)).value
    df = df.rename(columns={0: 'flux_normalized'})
    df = df[['Days', 'flux_normalized']]  

    # saving csv
    df.to_csv("lc_dummy.csv", index=False)

    # plotting
    ax = df.plot(x="Days", y="flux_normalized", alpha=0.5, linestyle='--', color ="blue")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalized Flux")
    ax.set_title("Lightcurve from PNG frames")
    plt.tight_layout()
    plt.savefig("flux_plot.png", dpi=600)
    plt.show()

# ---- Función de animación corregida ----
def animate(i, points, base_intensity, ax_sphere, elev, azim, total_frames, vmin, vmax):
    ax_sphere.clear()
    ax_sphere.set_axis_off()
    ax_sphere.view_init(elev=elev, azim=azim)
    
    # copying the texture
    intensity = np.copy(base_intensity)
    
    # adding several spots
    for spot in spots:
        # Calcular nueva posición de la mancha
        theta_mov = spot['theta'] + spot['angular_velocity'] * i * cadence_time.to(u.day)
        
        # Calculating position of the spot
        spot_x = r_val * np.sin(spot['phi']) * np.cos(theta_mov.value)
        spot_y = r_val * np.sin(spot['phi']) * np.sin(theta_mov.value)
        spot_z = r_val * np.cos(spot['phi'])
        spot_center = np.array([spot_x, spot_y, spot_z])
        
        # creating a mask
        mask = spot_mask_geodesic(points[:, 0], points[:, 1], points[:, 2], 
                                 spot_center, spot['radius'])
        intensity *= (1 -  mask)  # Reducción de intensidad en manchas
    
    # scatter plot
    sc = ax_sphere.scatter(
        points[:, 0], points[:, 1], points[:, 2], 
        c=intensity, 
        cmap='gray', 
        s=1, 
        alpha=0.9,
        vmin=vmin,  # Mínimo fijo
        vmax=vmax   # Máximo fijo
    )
    
    # Configurar límites de la esfera
    max_range = r_val * 1.1
    ax_sphere.set_xlim(-max_range, max_range)
    ax_sphere.set_ylim(-max_range, max_range)
    ax_sphere.set_zlim(-max_range, max_range)
    
    print(f"Procesando frame {i+1}/{total_frames}", end='\r')
    plt.savefig(f"frames/frame_{i:03d}.png", dpi=150, bbox_inches='tight')
    return None

# ---- main ----
if __name__ == '__main__':
    # stellar parameter
    r_val = 1.0
    n_points = 90000
    constant = 0.8  # limb darkening coefficients
    rotation_period = 1 * u.day
    
    # Point of view
    elev = 0
    azim = 0
    
    # List of spots
    spots = []
    
    #adding spots
    add_spots(-30, 0, 0.3)      
    add_spots(60, 90, 12)     
    add_spots(0, 0, 10)  
    
    # base lines time parameter
    observing_baseline_days = 1 * u.day
    cadence_time = 30 * u.minute
    total_frames = int((observing_baseline_days / cadence_time).decompose().value)
    
    # spherical grid with fibonacci points
    print("Generate spherical grid...")
    phi, theta = fibonacci_sphere(n_points)
    x, y, z = cartesian_from_spherical(phi, theta)
    points = np.vstack([x, y, z]).T
    
    # Calculate point of view
    elev_rad = np.deg2rad(elev) #elevation of point of view
    azim_rad = np.deg2rad(azim)#azimut of point of view
    
    v_x = np.cos(elev_rad) * np.cos(azim_rad)
    v_y = np.cos(elev_rad) * np.sin(azim_rad)
    v_z = np.sin(elev_rad)
    
    # rearrange of calculing mu parameter for limb darkening
    mu = (points[:, 0] * v_x + points[:, 1] * v_y + points[:, 2] * v_z) / r_val
    mu = np.clip(mu, 0, 1)
    base_intensity = limbdarkening(constant, mu)# applying to the texture
    
    #    Calculates the extreme values of the base intensity: 
    #vmin: Minimum value of intensity in the whole star.
    #vmax: Maximum value of intensity over the whole star.

   # Defines a reference range for color mapping that will be used consistently across all frames.
    vmin = np.min(base_intensity)
    vmax = np.max(base_intensity)
    
    # background configurations
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax_sphere = fig.add_subplot(111, projection='3d') 
    ax_sphere.set_axis_off()
    ax_sphere.set_box_aspect([1, 1, 1])
    
    # Límits
    max_range = r_val * 1.1
    ax_sphere.set_xlim(-max_range, max_range)
    ax_sphere.set_ylim(-max_range, max_range)
    ax_sphere.set_zlim(-max_range, max_range)
    
    # Generating animation
    print("start render...")
    for i in range(total_frames):
        animate(i, points, base_intensity, ax_sphere, elev, azim, total_frames, vmin, vmax)
    
    # Create gif an light curve
    print("\nCreando GIF...")
    gif(input_pattern="frames/frame_%03d.png", output_gif="lc_uniform.gif", framerate=18)
    print("Generating ligthcurve..")
    flux_plot()
