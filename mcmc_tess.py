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
    p0, _, _ = sampler.run_mcmc(p0, 3)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)


        # Diagnósticos

    return sampler, pos, prob, state

def plot_simple_traces(sampler, labels):
    """Gráficas simples de evolución de parámetros"""
    nwalkers, niter, ndim = sampler.chain.shape
    
    # 1. Evolución de cada parámetro
    fig, axes = plt.subplots(ndim + 1, 1, figsize=(10, 2*(ndim+1)))
    
    for i in range(ndim):
        ax = axes[i]
        for j in range(min(nwalkers, 10)):  # Solo 10 walkers para claridad
            ax.plot(sampler.chain[j, :, i], alpha=0.5, lw=0.8)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    
    # 2. Evolución del likelihood
    axes[-1].plot(np.mean(sampler.lnprobability, axis=0), 'b-', lw=2, label='Promedio')
    axes[-1].fill_between(range(niter),
                         np.percentile(sampler.lnprobability, 25, axis=0),
                         np.percentile(sampler.lnprobability, 75, axis=0),
                         alpha=0.3, label='Rango 25-75%')
    axes[-1].set_ylabel('ln(Likelihood)')
    axes[-1].set_xlabel('Iteración')
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)
    
    plt.suptitle('Evolución de Parámetros y Likelihood')
    plt.tight_layout()
    plt.savefig('traces_simples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig
    
def plot_walker_positions(sampler, labels):
    """Muestra la posición inicial vs final de los walkers"""
    nwalkers, niter, ndim = sampler.chain.shape
    
    fig, axes = plt.subplots(1, ndim, figsize=(4*ndim, 4))
    
    if ndim == 1:
        axes = [axes]
    
    for i in range(ndim):
        ax = axes[i]
        # Posiciones iniciales
        initial = sampler.chain[:, 0, i]
        # Posiciones finales
        final = sampler.chain[:, -1, i]
        
        ax.scatter([0]*nwalkers, initial, alpha=0.6, label='Inicial', s=50)
        ax.scatter([1]*nwalkers, final, alpha=0.6, label='Final', s=50)
        
        # Conectar puntos del mismo walker
        for w in range(min(nwalkers, 20)):
            ax.plot([0, 1], [initial[w], final[w]], 'gray', alpha=0.2, lw=0.5)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Inicio', 'Fin'])
        ax.set_ylabel(labels[i])
        ax.set_title(f'Movimiento {labels[i]}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('movimiento_walkers.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_simple_report(sampler, initial_params, labels):
    """Crea un reporte simple de la simulación"""
    flat_samples = sampler.flatchain
    flat_lnprob = sampler.flatlnprobability
    
    # Encontrar mejor muestra
    best_idx = np.argmax(flat_lnprob)
    best_params = flat_samples[best_idx]
    best_lnprob = flat_lnprob[best_idx]
    
    print("\n" + "="*50)
    print("REPORTE DE SIMULACIÓN MCMC (PRUEBA)")
    print("="*50)
    
    print(f"\n📊 Configuración:")
    print(f"   • Walkers: {sampler.chain.shape[0]}")
    print(f"   • Iteraciones: {sampler.chain.shape[1]}")
    print(f"   • Parámetros: {sampler.chain.shape[2]}")
    print(f"   • Total muestras: {len(flat_samples)}")
    
    print(f"\n🎯 Parámetros iniciales (MSE):")
    for i, label in enumerate(labels):
        print(f"   • {label}: {initial_params[i]:.3f}")
    
    print(f"\n🏆 Mejor ajuste encontrado:")
    for i, label in enumerate(labels):
        print(f"   • {label}: {best_params[i]:.3f}")
    print(f"   • ln(Likelihood): {best_lnprob:.2f}")
    
    print(f"\n📈 Rango explorado:")
    for i, label in enumerate(labels):
        samples = flat_samples[:, i]
        print(f"   • {label}: [{samples.min():.3f}, {samples.max():.3f}]")
    
    print(f"\n💾 Archivos generados:")
    print("   1. traces_simples.png - Evolución de parámetros")
    print("   2. movimiento_walkers.png - Movimiento de walkers")
    print("   3. corner_plot_simple.png - Distribuciones")
    
    # Guardar resultados en un archivo de texto
    with open('reporte_mcmc.txt', 'w') as f:
        f.write(f"Reporte MCMC - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write(f"Mejores parámetros:\n")
        for i, label in enumerate(labels):
            f.write(f"{label}: {best_params[i]:.6f}\n")
        f.write(f"ln(L): {best_lnprob:.2f}\n")
    
    return best_params, best_lnprob



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
    niter = 4
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


    # 1. Gráficas de evolución
    print("\nGenerando gráficas de evolución...")
    plot_simple_traces(sampler, labels)
    
    # 2. Movimiento de walkers
    print("Generando gráfica de movimiento...")
    plot_walker_positions(sampler, labels)
    
    # 3. Reporte simple
    print("Generando reporte...")
    best_params, best_lnprob = create_simple_report(sampler, initial, labels)
    
    # 4. Corner plot simple
    print("Generando corner plot...")
    flat_samples = sampler.flatchain
    
    # Tomar solo las últimas 5 iteraciones para el corner plot (si hay suficientes)
    if niter > 5:
        last_samples = sampler.chain[:, -5:, :].reshape(-1, ndim)
    else:
        last_samples = flat_samples
    
 # Usar una paleta de colores más profesional
    fig = corner.corner(
        last_samples,
        labels=labels,
        show_titles=True,
        title_kwargs={"fontsize": 10},
        title_fmt='.3f',
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=True,
        fill_contours=True,
        levels=[0.68, 0.95],  # Agregar nivel 95%
        color='#2E86AB',      # Azul profesional
        contour_kwargs={"colors": ['#2E86AB', '#A23B72']},  # Dos colores para contornos
        contourf_kwargs={"alpha": 0.6},  # Transparencia
        hist_kwargs={"color": "#2E86AB", "alpha": 0.7, "density": True},
        smooth=0.8,  # Suavizado
        range=[(p_min, p_max) for p_min, p_max in zip(last_samples.min(axis=0), last_samples.max(axis=0))],
        bins=20
    )
    
        
    # Marcar mejor ajuste
    corner.overplot_lines(fig, best_params, color='#F18F01', linewidth=2)
    corner.overplot_points(fig, best_params[None], color='#F18F01', 
                      marker='*', markersize=12, markeredgecolor='black')
    
    plt.suptitle(f'MCMC Test - {niter} iteraciones', fontsize=14)
    plt.savefig('corner_plot_simple.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Simulación completada!")
    print(f"📁 Resultados guardados en:")
    print(f"   • traces_simples.png")
    print(f"   • movimiento_walkers.png") 
    print(f"   • corner_plot_simple.png")
    print(f"   • reporte_mcmc.txt")
    
    # Pregunta simple de verificación
    print(f"\n❓ ¿Los walkers se están moviendo? (revisa movimiento_walkers.png)")
    print(f"❓ ¿El likelihood aumenta? (revisa traces_simples.png)")
    print(f"❓ ¿Los parámetros tienen distribución razonable? (revisa corner_plot_simple.png)")

##########
#########
##########
    # 5. NUEVA GRÁFICA: Likelihood vs cada parámetro (parámetros marginales)
    print("Generando gráfica Likelihood vs Parámetros...")
    fig, axes = plt.subplots(1, ndim, figsize=(4*ndim, 4))
    
    if ndim == 1:
        axes = [axes]
    
    for i in range(ndim):
        ax = axes[i]
        
        # Tomar una muestra aleatoria para no saturar (máx 1000 puntos)
        n_points_plot = min(1000, len(flat_samples))
        indices = np.random.choice(len(flat_samples), n_points_plot, replace=False)
        
        # Scatter plot: parámetro vs likelihood
        scatter = ax.scatter(flat_samples[indices, i], 
                            flat_lnprob[indices],
                            c=flat_lnprob[indices],
                            cmap='viridis',
                            alpha=0.6,
                            s=20,
                            edgecolors='none')
        
        # Línea suavizada (moving average)
        # Ordenar por valor del parámetro para la línea
        sorted_idx = np.argsort(flat_samples[indices, i])
        param_sorted = flat_samples[indices, i][sorted_idx]
        lnL_sorted = flat_lnprob[indices][sorted_idx]
        
        # Suavizado con ventana móvil
        window = max(1, n_points_plot // 20)
        if window > 1:
            kernel = np.ones(window) / window
            lnL_smooth = np.convolve(lnL_sorted, kernel, mode='same')
            ax.plot(param_sorted, lnL_smooth, 'r-', linewidth=2, alpha=0.8, 
                    label='Suavizado')
        
        # Marcar el máximo
        ax.axvline(best_params[i], color='red', linestyle='--', alpha=0.8,
                  label=f'Máximo: {best_params[i]:.2f}')
        
        ax.set_xlabel(labels[i], fontsize=12)
        ax.set_ylabel('ln(Likelihood)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        
        # Colorbar para la primera gráfica
        if i == 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('ln(Likelihood)', fontsize=10)
    
    plt.suptitle('Likelihood vs Parámetros Individuales', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig('likelihood_vs_params.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 6. Gráfica de evolución del likelihood para cada walker
    print("Generando gráfica de evolución por walker...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
    
    # Top: Evolución de likelihood por walker
    ax1 = axes[0]
    for w in range(min(nwalkers, 15)):  # Mostrar máximo 15 walkers
        ax1.plot(sampler.lnprobability[w, :], 
                 alpha=0.7, 
                 linewidth=0.8,
                 label=f'Walker {w}' if w < 5 else None)  # Solo etiquetar primeros 5
    
    ax1.set_ylabel('ln(Likelihood)', fontsize=12)
    ax1.set_title('Evolución del Likelihood por Walker', fontsize=14)
    ax1.grid(True, alpha=0.3)
    if nwalkers <= 5:
        ax1.legend(loc='lower right', fontsize=9)
    
    # Bottom: Promedio y desviación
    ax2 = axes[1]
    mean_lnprob = np.mean(sampler.lnprobability, axis=0)
    std_lnprob = np.std(sampler.lnprobability, axis=0)
    
    ax2.plot(mean_lnprob, 'b-', linewidth=2, label='Promedio')
    ax2.fill_between(range(niter), 
                     mean_lnprob - std_lnprob,
                     mean_lnprob + std_lnprob,
                     alpha=0.3, color='blue', label='±1σ')
    
    # Línea horizontal en el máximo
    max_lnprob = np.max(sampler.lnprobability[:, -1])
    ax2.axhline(max_lnprob, color='red', linestyle='--', alpha=0.7,
               label=f'Máximo final: {max_lnprob:.1f}')
    
    ax2.set_xlabel('Iteración', fontsize=12)
    ax2.set_ylabel('ln(L) Promedio', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('evolucion_likelihood.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Actualizar reporte para incluir info sobre el likelihood
    print("\n📊 INFORMACIÓN SOBRE EL LIKELIHOOD:")
    print("="*40)
    print("⚠️  El ln(Likelihood) SIEMPRE es negativo porque:")
    print("   • ln(probabilidad) donde probabilidad ∈ [0,1]")
    print("   • ln(x) < 0 cuando x < 1")
    print("\n✅ Lo importante es que:")
    print(f"   • ln(L) inicial ≈ {sampler.lnprobability[:, 0].mean():.1f}")
    print(f"   • ln(L) final   ≈ {sampler.lnprobability[:, -1].mean():.1f}")
    print(f"   • Mejoramiento: {sampler.lnprobability[:, -1].mean() - sampler.lnprobability[:, 0].mean():.1f}")
    print("\n🎯 Valores típicos:")
    print("   • ln(L) > -100: Ajuste excelente")
    print("   • -100 a -1000: Buen ajuste")
    print("   • -1000 a -10000: Ajuste moderado (común con muchos datos)")
    print(f"   • Tu valor: {best_lnprob:.1f} → {'Ajuste esperado para datos reales' if best_lnprob < -10000 else '¡Excelente ajuste!'}")
        
      