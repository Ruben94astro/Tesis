
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import R_sun
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.preprocessing import normalize
from line_profiler import profile

#---->Functions
@profile
def convertion_radians(degree):
    ''' convertions degree to radians'''
    return np.deg2rad(degree)
    
@profile
def limbdarkening(mu, u):
    '''Limb darkening lineal model'''
    return (1 - u * (1 - mu))
    
@profile
def spot_theta(days, initial_rad, spot_phi_rad, omega_eq, delta_omega, frame, total_frames):
    """
    Function to calculate the motion of the spots and the differential rotation
    Initial position of the spots.
    Parameters:
    -days
    - initial_deg: initial position of spot.
    - spot_phi_rad: latitud of spot(radians)
    - omega_eq: angular velocity.
    - delta_omega: difference between velocity in pole and equator.
    - frame: actual frame.
    - total_frames:.
    Returns:
    - theta in radians.
    """  
    delta_t = days/total_frame
    time = frame * delta_t

    ##Differential Rotation
    omega_phi = omega_eq - delta_omega * np.sin(spot_phi_rad)**2#---->sunlike star
    delta_theta_d = omega_phi * time
    return initial_rad + delta_theta_d



@profile
def spot_mask(theta_mesh, spot_theta_rad, phi_mesh, spot_phi_rad, spot_radius_rad):
    '''function for creating spot mask, that could change the size and shape of the spot
    using a gaussian function
    '''
    delta_theta = np.arccos(np.cos(theta_mesh - spot_theta_rad))
    delta_phi = np.abs(phi_mesh - spot_phi_rad)
    distance_squared = delta_theta**2 + delta_phi**2
    sigma_squared = (spot_radius_rad / 2)**2
    return np.exp(-distance_squared / (2 * sigma_squared))

@profile
def flux_plot(flux):
    ''' Function take a list normalizing the flux, converting the list in a csv file and rename the columns
    and return a plot    
    '''
    flux_norm = normalize([flux], norm="max")[0]
    df = pd.DataFrame(flux_norm)
    df.to_csv("lc_high.csv")
    lc = pd.read_csv("lc_high.csv")
    lc = lc.rename(columns={'Unnamed: 0': 'Days', '0': 'flux_normalized'})
    lc.to_csv('lc_high_name.csv')
    return lc.plot(x="Days", y="flux_normalized", alpha=0.5)

# === Animation function ===
@profile
def animate(i, gray_texture, mu, spot_theta_rad, spot_phi_rad, spot_radius_rad,
            theta_mesh, phi_mesh, x, y, z, ax_sphere, elev, azim, ax_curve, line_curve, fluxes,
            days, omega_eq, delta_omega, total_frame):
    """ Function where is created the motion and call functions"""
    
    ax_sphere.clear()
    ax_sphere.set_axis_off()
    ax_sphere.view_init(elev=elev, azim=azim)

    texture = np.copy(gray_texture)
    spot_theta_motion = spot_theta(days, spot_theta_rad, spot_phi_rad, omega_eq, delta_omega, i, total_frame)
    spot_mask_motion = spot_mask(theta_mesh, spot_theta_motion, phi_mesh, spot_phi_rad, spot_radius_rad)
    
    #applying texture to the spot
    texture *= (1 - 0.9 * spot_mask_motion)
    
    #plotting the surface
    surf = ax_sphere.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.gray(texture), shade=False)
    
    #taking only visible part
    visible = mu > 0
    total_flux = np.sum(texture[visible])
    fluxes.append(total_flux)

    #Normalized fluz for plot
    fluxes_normalized = normalize([fluxes], norm="max")[0]
    line_curve.set_data(np.arange(len(fluxes))*days/total_frame, fluxes_normalized)

    print(f"Procesando frame {i+1}/{total_frame}", end='\r')
    return [surf, line_curve]

#------->main
if __name__ == '__main__':
    # Parameters
    r = 1 * u.R_sun#------->solar like radii
    res = 200#--------->resolution
    constant = 0.8#---------->limb darkening coefficient
    spot_theta_deg = 0#------->longitude
    spot_phi_deg = 45#--------->latitude
    spot_radius_deg = 25#------->radii spot
    #parameters of motion
    omega_eq = 2.9 * np.pi#------>equator angular velocity
    delta_omega = 0.5 * np.pi#----->difference between equator and pole velocity
    days = 10#----->days that you want to put
    total_frame = 240#-------->how many frames do you wants
    elev = 30#-------->elevation angle
    azim = 0#-------->doesn't affect

    # Conversion parameters
    spot_phi_rad = convertion_radians(spot_phi_deg)
    spot_radius_rad = convertion_radians(spot_radius_deg)
    spot_theta_rad = convertion_radians(spot_theta_deg)
    elev_rad = convertion_radians(elev)
    azim_rad = convertion_radians(azim)

    # Meshgrid
    theta = np.linspace(0, 2*np.pi, res)
    phi = np.linspace(0, np.pi, res)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)
    x = r * np.sin(phi_mesh) * np.cos(theta_mesh)
    y = r * np.sin(phi_mesh) * np.sin(theta_mesh)
    z = r * np.cos(phi_mesh)

    # Point of view of limb darkening
    v_x = np.cos(elev_rad) * np.cos(azim_rad)
    v_y = np.cos(elev_rad) * np.sin(azim_rad)
    v_z = np.sin(elev_rad)
    mu = (x * v_x + y * v_y + z * v_z) / r
    mu = np.clip(mu, 0, 1)

    # Texture for limb darkening
    gray_texture = np.ones((res, res))
    gray_texture *= limbdarkening(mu, constant)

    # Plot figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 6))
    ax_sphere = fig.add_subplot(121, projection='3d')
    ax_curve = fig.add_subplot(122)
    ax_sphere.set_box_aspect((0.98, 1, 0.95))
    ax_sphere.set_axis_off()
    ax_curve.set_title('Lightcurve', color='white')
    ax_curve.set_xlabel('Days')
    ax_curve.set_ylabel('Normalized flux')
    ax_curve.set_xlim(0, days)
    ax_curve.set_ylim(0.7, 1.05)
    ax_curve.set_facecolor('black')
    line_curve, = ax_curve.plot([], [], color='orange')
    
    #list for saving fluxes
    fluxes = []

    ani = animation.FuncAnimation(
        fig, animate, frames=total_frame, interval=60, blit=False, repeat=False,
        fargs=(gray_texture, mu, spot_theta_rad, spot_phi_rad, spot_radius_rad,
               theta_mesh, phi_mesh, x, y, z, ax_sphere, elev, azim, ax_curve,
               line_curve, fluxes, days, omega_eq, delta_omega, total_frame)
    )

    ani.save('spot_high_period_def_corrected.gif', writer='ffmpeg', fps=20)
    plt.show()
