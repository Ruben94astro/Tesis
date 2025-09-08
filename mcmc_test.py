%matplotlib inline
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
import gc
import matplotlib
import seaborn as sns
from sklearn.metrics import mean_squared_error
import emcee



# 1. Read Lc

observed_file = "test.csv"
df_obs = pd.read_csv(observed_file)
obs_flux = df_obs["flux_normalized"].values

# ==========================================
# 2. run simulation
# ==========================================
def simulate_curve(lat, lon, r, output_file="simulated.csv"):




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
    
    def quadratic(u1, u2, mu):
        return 1-u1*(1-mu)-u2*(1-mu)**2
        
    
    def add_spots(latitude_deg, longitude_deg, radii_deg):
        colatitude_rad = np.deg2rad(90 - latitude_deg)
        longitude_rad = np.deg2rad(longitude_deg)
        radii_rad = np.deg2rad(radii_deg)
        ang_vel = spot_theta(rotation_period, colatitude_rad, 0.4)  
        spots.append({
            'theta': longitude_rad * u.rad,
            'phi': colatitude_rad,
            'radius': radii_rad,
            'angular_velocity': ang_vel
        })
    
    def gif(input_pattern="frames/frame_%05d.png", output_gif="output.gif", 
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
        #flux_norm.append(flux_total / fluxes[i])
        flux_norm = normalize([fluxes], norm="max")[0]
        #flux_norm = np.array(fluxes)/np.max(np.array(fluxes))
        df = pd.DataFrame(flux_norm)
    
        # Creating columns
        df.index.name = 'Frame'
        df.reset_index(inplace=True)
        #changing frames for days
        df['Days'] = df['Frame'] *(cadence_time.to(u.day)).value
        df = df.rename(columns={0: 'flux_normalized'})
        df = df[['Days', 'flux_normalized']]  
    
        # saving csv
        #df.to_csv(f'lat_{(90 - np.rad2deg(spots[0]["phi"])):.1f}_lon{(np.rad2deg(spots[0]["theta"]).value):.1f}_radii{np.rad2deg(spots[0]["radius"]):.1f}.csv', index=False)
        df.to_csv(".csv", index=False)
        # plotting
        
        ax = df.plot(x="Days", y="flux_normalized", alpha=0.5, linestyle='--', color ="k")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Normalized Flux")
        ax.set_title("Lightcurve from PNG frames")
        plt.style.use('default')
        plt.tight_layout()
        plt.savefig(f'lat_{(90 - np.rad2deg(spots[0]["phi"])):.1f}_lon{(np.rad2deg(spots[0]["theta"]).value):.1f}_radii{np.rad2deg(spots[0]["radius"]):.1f}.png', dpi=600)
        plt.show()
    
    # ----animation function
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
            #c=intensity,
            c=np.clip(intensity, 0, 1),
            cmap='gray', 
            s=1, 
            alpha=1,
            vmin=0,  # Mínimo fijo
            vmax=1.0   # Máximo fijo
        )
        
        # Configurar límites de la esfera
        max_range = r_val * 1.1
        ax_sphere.set_xlim(-max_range, max_range)
        ax_sphere.set_ylim(-max_range, max_range)
        ax_sphere.set_zlim(-max_range, max_range)
        
        print(f"Procesando frame {i+1}/{total_frames}", end='\r')
        plt.savefig(f"frames/frame_{i:05d}.png", dpi=300, bbox_inches='tight')
        
        return None
    
    # ---- main ----
    if __name__ == '__main__':
        # stellar parameter
        #lat, long, radii = theta_vec
        r_val = 1.0
        n_points = 10000
        u1 = 0.4
        u2 =0.3# limb darkening coefficients
        rotation_period = 1.0 * u.day
        
        # Point of view
        elev = 0
        azim = 0
        
        # List of spots
        spots = []
        
        #adding spots
        add_spots(lat, lon, r)      
        #add_spots(-50, 0, 20.5)     
        #add_spots(0, 0, 20)  
        #add_spots(50, 240, 10.5) 
        
        # base lines time parameter
        observing_baseline_days = 1 * u.day
        cadence_time = 50 * u.minute
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
        
        # rearrange of calculating mu parameter for limb darkening
        mu = (points[:, 0] * v_x + points[:, 1] * v_y + points[:, 2] * v_z) / r_val
        mu = np.clip(mu, 0, 1)
        #base_intensity = limbdarkening(constant, mu)# applying to the texture
        base_intensity = quadratic(u1,u2,mu) 
        #    Calculates the extreme values of the base intensity: 
        #vmin: Minimum value of intensity in the whole star.
        #vmax: Maximum value of intensity over the whole star.
        vmin=  0.0
        vmax=  1.0
    
       # Defines a reference range for color mapping that will be used consistently across all frames.
     
        
        # background configurations
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 8))
        ax_sphere = fig.add_subplot(111, projection='3d') 
        ax_sphere.set_axis_off()
        ax_sphere.set_box_aspect([1, 1, 1])
        
        #Límits
        max_range = r_val * 1.1
        ax_sphere.set_xlim(-max_range, max_range)
        ax_sphere.set_ylim(-max_range, max_range)
        ax_sphere.set_zlim(-max_range, max_range)
        
        # Generating animation
        print("start render...")
        for i in range(total_frames):
            animate(i, points, base_intensity, ax_sphere,  elev, azim, total_frames, vmin, vmax)
    
        
        # Create gif an light curve
        #print("\nCreando GIF...")
        #gif(input_pattern="frames/frame_%05d.png", output_gif=f"period_{rotation_period}_points{n_points}_obs{observing_baseline_days}_cadence{cadence_time}_nspots{len(spots)}.gif", framerate=15)
        print("Generating ligthcurve..")
        plt.style.use('default')
        flux_plot()
    


    return output_file


# 3. Log-likelihood (based en MSE)

def lnlike(params, obs_flux):
    lat, lon, r = params
    try:
        sim_file = simulate_curve(lat, lon, r)
        df_sim = pd.read_csv(sim_file)
        sim_flux = df_sim["flux_normalized"].values
        if np.any(np.isnan(sim_flux)):
            return -np.inf  # Evita valores NaN
        mse = mean_squared_error(obs_flux, sim_flux)
        return -0.5 * mse
    except Exception as e:
        print(f"Error en simulación: {e}")
        return -np.inf  # Retorna -inf si hay error

# ==========================================
# 4. Priors
# ==========================================
def lnprior(params):
    lat, lon, r = params
    # Ejemplo: lat entre -90 y 90, lon entre 0 y 360, radio positivo
    if -90 <= lat <= 90 and 0 <= lon < 360 and 0.1 <= r <= 50:
        return 0.0  # log(1)
    return -np.inf  # imposible

# ==========================================
# 5. Log-probabilidad total
# ==========================================
def lnprob(params, obs_flux):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, obs_flux)


# 6.  walkers

ndim = 3
nwalkers = 16
niter = 200  

# initial points
params_init = np.array([0.0, 0.0, 5.0])  # (lat, lon, r)
p0 = [params_init + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]

# ==========================================
# 7. run sampler

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(obs_flux,))
sampler.run_mcmc(p0, niter, progress=True)

# results

chain = sampler.get_chain()
print("Chain shape:", chain.shape)  # (niter, nwalkers, ndim)

# Graficar la evolución de cada parámetro
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ["Latitud", "Longitud", "Radio"]
for i in range(ndim):
    axes[i].plot(chain[:, :, i], "k", alpha=0.3)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("Step")
plt.show()
