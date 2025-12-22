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
#import seaborn as sns
from sklearn.metrics import mean_squared_error
import emcee
import corner
from concurrent.futures import ProcessPoolExecutor
import tqdm  # opcional, para barra de progreso
from multiprocessing import Pool
from datetime import datetime




os.makedirs("frames", exist_ok=True)
# --- Esto va al inicio del archivo, nivel superior ---
def animate(i, points, base_intensity, elev, azim,
                  total_frames, vmin, vmax, spots, r_val, cadence_time):
            # background configurations
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax_sphere = fig.add_subplot(111, projection='3d') 
    ax_sphere.set_axis_off()
    ax_sphere.set_box_aspect([1, 1, 1])
    ax_sphere.view_init(elev=elev, azim=azim)
    

    
    #fig = plt.figure(figsize=(6, 6))
    #ax_sphere = fig.add_subplot(111, projection='3d')
    #ax_sphere.set_axis_off()
    #ax_sphere.set_box_aspect([1, 1, 1])
    #ax_sphere.view_init(elev=elev, azim=azim)

    intensity = np.copy(base_intensity)

    for spot in spots:
        theta_mov = spot['theta'] + spot['angular_velocity'] * -1*i * cadence_time.to(u.day)
        spot_x = r_val * np.sin(spot['phi']) * np.cos(theta_mov.value)
        spot_y = r_val * np.sin(spot['phi']) * np.sin(theta_mov.value)
        spot_z = r_val * np.cos(spot['phi'])
        spot_center = np.array([spot_x, spot_y, spot_z])

        mask = spot_mask_geodesic(points[:,0], points[:,1], points[:,2],
                                  spot_center, spot['radius'])
        contrast = 0.48   # c_TESS calculado físicamente
        intensity *= (1 - mask + contrast * mask)

    ax_sphere.scatter(points[:,0], points[:,1], points[:,2],
                      c=np.clip(intensity,0,1), cmap='gray', s=1,
                      vmin=vmin, vmax=vmax)




    max_range = r_val * 1.1
    ax_sphere.set_xlim(-max_range, max_range)
    ax_sphere.set_ylim(-max_range, max_range)
    ax_sphere.set_zlim(-max_range, max_range)

    os.makedirs("frames", exist_ok=True)
    plt.savefig(f"frames/frame_{i:05d}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    return 


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
    pos_norm = positions / np.clip(norms, 1e-10, None)     
    dot_product = np.sum(pos_norm * center_norm, axis=-1) 
    dot_product = np.clip(dot_product, -1, 1) 
    
    # central_angle is the geodesic distance (in radians) betwe.
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
    #return spots
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

def flux_plot(theta_vec):
    '''
    Function take a list normalizing the flux, converting the list in a 
csv file and rename the columns
    and return a plot 

    '''
    #lat,lon,radii = theta_vec
    lat, lon, radii = theta_vec
    frame_files = sorted(glob.glob("frames/frame_*.png"))
    fluxes = []

    with Pool() as pool:
        fluxes = pool.map(compute_flux, frame_files)

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
    df.to_csv(f'lat_{lat}_lon_{lon}_radii{radii}.csv', index=False)
 
    
    #df.to_csv("test.csv", index=False)
    # plotting
    

    return df["flux_normalized"], df["Days"]

#------pruebas
def compute_flux(filename):
    img = Image.open(filename).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return np.sum(img_array)

# --- paralelización ---


# ---- Función de animación corregida ----



def run_parallel_frames(points, base_intensity, elev, azim,
                        total_frames, vmin, vmax, spots, r_val, cadence_time,
                        n_workers=198):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(animate, i, points, base_intensity,
                            elev, azim, total_frames, vmin, vmax,
                            spots, r_val, cadence_time)
            for i in range(total_frames)

        ]
        for f in tqdm.tqdm(futures):
            f.result()



 # Generating animation
def star_animate(theta_vec):
    
    #lat,lon,radii = theta_vec
    lat,lon,radii = theta_vec
    add_spots(lat, lon, radii)
        # spherical grid with fibonacci points
    print("Generate spherical grid")
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
    
 
    

    print("start render...")
 # usa el # de núcleos que tengas

        #animate(i, points, base_intensity, ax_sphere,  elev, azim, total_frames, vmin, vmax)
    # Create gif an light curve
    #print("\nCreando GIF...")
    #gif(input_pattern="frames/frame_%05d.png", output_gif=f"period_{rotation_period}_points{n_points}_obs{observing_baseline_days}_cadence{cadence_time}_nspots{len(spots)}.gif", framerate=15)
    #to verify if is mcmc is working 
    #print(theta_vec)
    run_parallel_frames(points, base_intensity, elev, azim,
                    total_frames, vmin, vmax, spots, r_val, cadence_time,
                    n_workers=198) 
    
    plt.style.use('default')
    spots.clear()
    plt.close(fig) 
 
    return flux_plot(theta_vec)

def load_tess():
# load tess light_curves
    csv1 = "tess_curve.csv"
    #csv2 = f"lat_{float(vec[0])}_lon{float(vec[1])}_radii{float(vec[2])}.csv"
    
    df1 = pd.read_csv(csv1)
    #df2 = pd.read_csv(csv2)
    F = df1["flux_normalized"]*0.9961104871028598 #scale factor
    days =df1["Time"]
    F_error = df1["flux_error_normalized"]*0.9961104871028598 #scale factor
    return F, days, F_error

#  verification function to find the aprox parameters value, recibing the test light curve and days


def function_mse(flux, days):
    sim_folder = 'simulation'
    results = []
    
    for file in os.listdir(sim_folder):
        if file.endswith('.csv'):
    
            path = os.path.join(sim_folder, file)
            
            try:
                # Leer archivo
                sim_curve = pd.read_csv(path)
                sim_flux = sim_curve['flux_normalized'].values
                sim_time = sim_curve['Days'].values
                
                # Interpolar si es necesario
                if len(sim_flux) != len(flux):
                    print("different size")
                
                # Calcular MSE
                mse = np.mean((flux - sim_flux) ** 2)
                
                # EXTRAER PARÁMETROS - VERSIÓN SIMPLIFICADA
                # Formato: la_20lon_60radii9.csv
                
                # 1. Eliminar extensión y dividir por '_'
                name = file.replace('.csv', '')
                parts = name.split('_')
                

                
                # 2. Extraer longitud (segunda parte: "20lon")
                lon_part = parts[1]  # "20lon"

                lon = float(lon_part.replace('lon', ''))
                
                # 3. Extraer latitud y radio (tercera parte: "60radii9")
                radii_part = parts[2]  # "60radii9"
                
                # Buscar 'radii' en la cadena

                
                # Separar por 'radii'
                lat_str, rad_str = radii_part.split('radii')
                lat = float(lat_str)
                rad = float(rad_str)
                
        
                
                results.append({
                    'file': file,
                    'lat': lat,
                    'lon': lon,
                    'rad': rad,
                    'mse': mse,
                    'flux': sim_flux
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
    
    if not results:
        print("❌ No hay resultados!")
        return None
    
    # Ordenar y retornar
    results.sort(key=lambda x: x['mse'])
    top5 = results[:5]
    print("\nTop 5 likely light curve:\n")
    for i, r in enumerate(top5, 1):
        print(f"{i}. Archivo: {r['file']}, Lat: {r['lat']}, Lon: {r['lon']}, Rad:{r['rad']}, MSE: {r['mse']:.10f}")
    
    return top5[0]["lon"], top5[0]["lat"],top5[0]["rad"]

def lnlike(theta_vec, F, days, Ferr):
    flux_simulated, days_simulated = star_animate(theta_vec)

    
    # Verosimilitud gaussiana apropiada para MCMC
    chi2 = np.sum((F - flux_simulated)**2 / Ferr**2)
    return -0.5 * chi2

def lnprior(theta_vec):
    lat, lon, radii = theta_vec
    if (-90.0 <= lat <= 90.0 and 
        0.0 <= lon < 360.0 and  # o 0-360 según tu convención
        0.1 <= radii <= 15.0):
        return 0.0
    return -np.inf


def lnprob(theta_vec, data):
    days, F, Ferr = data
    lp = lnprior(theta_vec)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta_vec, F, days, Ferr)



def main(p0,nwalkers,niter,ndim,lnprob,data,save_chain=True):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,))

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 10)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

        # Diagnósticos
    print("Autocorrelation time:", sampler.get_autocorr_time())
    print("Acceptance fraction:", np.mean(sampler.acceptance_fraction))
    ###
        # Diagnósticos
    try:
        tau = sampler.get_autocorr_time()
        print(f"Autocorrelation time: {tau}")
        print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        print(f"Number of effective samples: {sampler.get_chain().shape[0] / np.mean(tau):.0f}")
    except:
        print("Could not compute autocorrelation time")
    
    if save_chain:
        # Guardar la cadena
        np.save('mcmc_chain.npy', sampler.get_chain())
        np.save('mcmc_log_prob.npy', sampler.get_log_prob())


    return sampler, pos, prob, state

def simple_autocorr(x, max_lag=100):
    n = len(x)
    if n < 10:
        return np.zeros(max_lag)
    
    x = x - np.mean(x)
    var = np.var(x)
    
    if var == 0:
        return np.zeros(max_lag)
    
    acf = np.zeros(max_lag)
    for lag in range(min(max_lag, n-1)):
        acf[lag] = np.sum(x[:n-lag] * x[lag:]) / (var * (n - lag))
    
    return acf


##analysis block
def generate_mcmc_report(sampler, data, param_names, best_params, true_params=None):
  


    days, flux, flux_err = data
    n_params = len(param_names)
    
    # Crear una figura grande con subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. TRAZAS DE CADENAS (Temporal evolution)
    for i in range(n_params):
        ax = plt.subplot(n_params + 3, 3, i*3 + 1)
        samples = sampler.get_chain()[:, :, i]
        for walker in range(min(20, samples.shape[1])):  # Mostrar solo 20 walkers para             
            ax.plot(samples[:, walker], alpha=0.3, linewidth=0.5)
            ax.set_ylabel(param_names[i])
            ax.set_xlabel('Iteración')
            ax.grid(True, alpha=0.3)
        
    # 2. DISTRIBUCIÓN POSTERIOR
    for i in range(n_params):
        ax = plt.subplot(n_params + 3, 3, i*3 + 2)
        flat_samples = sampler.get_chain()[:, :, i].flatten()
        ax.hist(flat_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
        
        # Calcular estadísticas
        mean_val = np.mean(flat_samples)
        median_val = np.median(flat_samples)
        std_val = np.std(flat_samples)
        
        # Percentiles
        q16, q50, q84 = np.percentile(flat_samples, [16, 50, 84])
        
        # Plot estadísticas
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
        if true_params is not None:
            ax.axvline(true_params[i], color='blue', linestyle='-', 
                      linewidth=2, label=f'True: {true_params[i]:.3f}')
        
        ax.set_xlabel(param_names[i])
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Añadir texto con estadísticas
        stats_text = f'σ = {std_val:.3f}\n16%: {q16:.3f}\n84%: {q84:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    #  3. DIAGNÓSTICOS DE CONVERGENCIA
    # ax_conv ocupará toda la última fila (3 columnas)
    ax_conv = plt.subplot2grid((n_params + 3, 3), (n_params, 0), colspan=3)
    mean_log_prob = np.mean(sampler.get_log_prob(), axis=1)
    ax_conv.plot(mean_log_prob, label='Mean log probability')
    ax_conv.set_xlabel('Iteración')
    ax_conv.set_ylabel('Log Probability')
    ax_conv.legend()
    ax_conv.grid(True, alpha=0.3)
    
    # 4. FUNCIÓN DE AUTOCORRELACIÓN
    if n_params <= 3:
        for i in range(n_params):
            ax_acf = plt.subplot(n_params + 3, 3, (n_params+1)*3 + i + 1)
            chain = sampler.get_chain()[:, :, i]
            # Calcular autocorrelación para cada walker y promediar
            max_lag = min(100, chain.shape[0] // 2)
            acf_mean = np.zeros(max_lag)
            for w in range(chain.shape[1]):
                acf = simple_autocorr(chain[:, w], max_lag=max_lag)
                acf_mean += acf[:max_lag]
            acf_mean /= chain.shape[1]
            
            ax_acf.plot(acf_mean)
            ax_acf.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax_acf.set_xlabel('Lag')
            ax_acf.set_ylabel(f'ACF ({param_names[i]})')
            ax_acf.grid(True, alpha=0.3)
    
    plt.suptitle('MCMC Analysis Report', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('mcmc_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def generate_statistical_summary(sampler, param_names, best_params):
    """
    Genera un resumen estadístico detallado.
    """
    summary = "\n" + "="*60 + "\n"
    summary += "MCMC STATISTICAL SUMMARY\n"
    summary += "="*60 + "\n\n"
    
    for i, name in enumerate(param_names):
        flat_samples = sampler.get_chain()[:, :, i].flatten()
        
        # Estadísticas básicas
        mean_val = np.mean(flat_samples)
        median_val = np.median(flat_samples)
        std_val = np.std(flat_samples)
        mad_val = np.median(np.abs(flat_samples - median_val))
        
        # Intervalos de credibilidad
        q5, q16, q50, q84, q95 = np.percentile(flat_samples, [5, 16, 50, 84, 95])
        
        # Mejor valor (máximo de probabilidad)
        best_val = best_params[i]
        
        summary += f"Parameter: {name}\n"
        summary += "-"*40 + "\n"
        summary += f"  Mean:        {mean_val:.4f} ± {std_val:.4f}\n"
        summary += f"  Median:      {median_val:.4f} (MAD: {mad_val:.4f})\n"
        summary += f"  Best fit:    {best_val:.4f}\n"
        summary += f"  68% CI:      [{q16:.4f}, {q84:.4f}]\n"
        summary += f"  95% CI:      [{q5:.4f}, {q95:.4f}]\n"
        summary += f"  R-hat (approx): {gelman_rubin(sampler, i):.3f}\n\n"
    
    # Información del sampler
    summary += "Sampler Information:\n"
    summary += "-"*40 + "\n"
    summary += f"Number of walkers:      {sampler.get_chain().shape[1]}\n"
    summary += f"Number of iterations:   {sampler.get_chain().shape[0]}\n"
    summary += f"Total samples:         {sampler.get_chain().size // len(param_names)}\n"
    summary += f"Mean acceptance rate:  {np.mean(sampler.acceptance_fraction):.3f}\n"
    
    try:
        tau = sampler.get_autocorr_time()
        summary += f"Autocorrelation time:  {tau}\n"
        summary += f"Effective samples:     {sampler.get_chain().shape[0] / np.mean(tau):.0f}\n"
    except:
        summary += "Autocorrelation time:  Could not compute\n"
    
    return summary

def gelman_rubin(sampler, param_idx):
    """
    Calcula el estadístico R-hat de Gelman-Rubin para diagnosticar convergencia.
    """
    chains = sampler.get_chain()[:, :, param_idx]
    n_iter, n_walkers = chains.shape
    
    # Mean of each chain
    chain_means = np.mean(chains, axis=0)
    
    # Mean of all chains
    overall_mean = np.mean(chains)
    
    # Between-chain variance
    B = n_iter / (n_walkers - 1) * np.sum((chain_means - overall_mean)**2)
    
    # Within-chain variance
    W = np.mean(np.var(chains, axis=0, ddof=1))
    
    # Variance estimate
    var_est = (n_iter - 1) / n_iter * W + B / n_iter
    
    # R-hat statistic
    Rhat = np.sqrt(var_est / W)
    
    return Rhat

def plot_corner_with_stats(samples, param_names, best_params):
    """
    Corner plot mejorado con estadísticas.
    """
    fig = corner.corner(samples, 
                       labels=param_names,
                       show_titles=True,
                       title_fmt='.3f',
                       quantiles=[0.16, 0.5, 0.84],
                       plot_datapoints=True,
                       fill_contours=True,
                       levels=[0.68, 0.95],
                       color='blue',
                       truths=best_params,
                       truth_color='red',
                       label_kwargs={'fontsize': 12})
    
    # Añadir información adicional
    fig.text(0.95, 0.95, f'N = {len(samples):,} samples',
             ha='right', va='top',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig('corner_plot_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def save_complete_report(sampler, data, param_names, best_params, config):
    """
    Guarda un informe completo en un archivo de texto.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open('mcmc_analysis_report.txt', 'w') as f:
        f.write(f"MCMC Analysis Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("="*60 + "\n\n")
        
        f.write("CONFIGURATION PARAMETERS:\n")
        f.write("-"*40 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write(generate_statistical_summary(sampler, param_names, best_params))
        
        f.write("\nCONVERGENCE DIAGNOSTICS:\n")
        f.write("-"*40 + "\n")
        f.write("R-hat statistics (should be < 1.1 for convergence):\n")
        for i, name in enumerate(param_names):
            Rhat = gelman_rubin(sampler, i)
            status = "✓ CONVERGED" if Rhat < 1.1 else "⚠ NEEDS MORE ITERATIONS"
            f.write(f"  {name}: {Rhat:.3f} - {status}\n")
    
    print(f"Report saved to 'mcmc_analysis_report.txt'")
## end analysis block

# ---- main ----
if __name__ == '__main__':
    # stellar parameter
    r_val = 1.0
    n_points = 65000
    u1 = 0.2076
    u2 =0.3849# limb darkening coefficients
    rotation_period = 0.5926 * u.day
    
    # Point of view
    elev = 32
    azim = 0
    
    # List of spots
    spots = []
  
    
    # base lines time parameter
    observing_baseline_days = 3 * u.day
    cadence_time = 2 * u.minute
    total_frames = int((observing_baseline_days / cadence_time).decompose().value)
    
    #load flux and days of test light curve
    flux_tess, days_tess,flux_tess_error =load_tess()
    print(f"Datos cargados: {len(flux_tess)} puntos")
    #parameters where we are selecting the top1 simulated lightcurve by MSE
    initial_params = function_mse(flux_tess, days_tess)
    print(f"Parámetros iniciales encontrados: {initial_params}")
   

    #flux_err = 0.001 * (np.max(flux_test) - np.min(flux_test)) 
    
    data = (days_tess,flux_tess,flux_tess_error)
    nwalkers = 6
    niter = 5
    initial = np.array(initial_params)
    
    #ndim = len(initial)
    ndim = len(initial)
    #p0 = np.array([initial + 2*np.random.randn(ndim) for i in range(nwalkers)])[:, None]
    p0 = [np.array(initial) +  2* np.random.randn(ndim) for i in range(nwalkers)]
    
    
    sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)

        # Obtener los mejores parámetros
    samples = sampler.flatchain
    best_params = samples[np.argmax(sampler.flatlnprobability)]
    
    labels = ['lat', 'lon','radii']
    
    # Crear una sola figura con los valores verdaderos
    fig = corner.corner(samples, 
                       show_titles=True, 
                       labels=labels, 
                       plot_datapoints=True, 
                       quantiles=[0.16, 0.5, 0.84],
                       truths=best_params,
                       truth_color='red')  # Puedes personalizar el color
    
    # Agregar texto con los valores en una posición que no se superponga
    # Usamos transform=fig.transFigure para coordenadas relativas a la figura completa
    fig.text(0.75, 0.75, f'Best params: {np.round(best_params, 2)}', 
             fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8),
             transform=fig.transFigure,  # Importante: usa coordenadas de figura
             ha='center')  # Alineación horizontal centrada
    
    # Ajustar diseño para evitar superposiciones
    plt.tight_layout()
    
    # Guardar la figura
    fig.savefig('corner_plot_with_truths.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    
        # 1. Reporte visual completo
    true_params = None 
    generate_mcmc_report(sampler, data, labels, best_params, true_params)
    
    # 2. Corner plot mejorado
    plot_corner_with_stats(samples, labels, best_params)
    
    
    # 3. Resumen estadístico en consola
    print(generate_statistical_summary(sampler, labels, best_params))
    
    # 4. Guardar reporte completo en archivo
    
    config = {
    'nwalkers': nwalkers,
    'niter': niter,
    'ndim': ndim,
    'initial_params': initial_params,
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    save_complete_report(sampler, data, labels, best_params, config)
    
    # 5. Plot adicional: Mejor ajuste vs datos
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Simular curva con mejores parámetros
    flux_best, days_best = star_animate(best_params)
    
    ax.errorbar(days_test, flux_test, yerr=flux_err, fmt='.', 
                alpha=0.5, label='Datos', markersize=2)
    ax.plot(days_best, flux_best, 'r-', linewidth=2, label='Mejor ajuste MCMC')
    
    # Añadir sombra para incertidumbre (muestreo de parámetros)
    n_samples_plot = 50
    for i in np.random.choice(len(samples), n_samples_plot):
        flux_sample, _ = star_animate(samples[i])
        ax.plot(days_best, flux_sample, 'gray', alpha=0.05)
    
    ax.set_xlabel('Días')
    ax.set_ylabel('Flujo Normalizado')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Mejor ajuste MCMC con incertidumbre')
    
    plt.savefig('best_fit_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print("Archivos generados:")
    print("1. mcmc_report.png - Reporte visual completo")
    print("2. corner_plot_enhanced.png - Corner plot con estadísticas")
    print("3. mcmc_analysis_report.txt - Reporte detallado en texto")
    print("4. best_fit_comparison.png - Comparación datos vs modelo")
    print("5. mcmc_chain.npy - Cadena completa (si save_chain=True)")


    
