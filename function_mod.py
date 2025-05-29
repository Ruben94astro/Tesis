#libraries
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import R_sun
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.preprocessing import normalize
import astropy.units as u
#functions 

# convertion function 
def convertion_radians(degree):
    return np.deg2rad(degree)

#function to calculate angular velocity    
def spot_theta(rotation_period, spot_colatitude, relative_shear):
    """
    Parameters:
    - rotation_period: rotation period of the sta
    - initial_latitude: latitud of spot(radians)
    - relative_shear: parameter between pole and equator
    Returns:
    - angular velocity.
    """
    latitude = np.pi / 2 - spot_colatitude #------->the spehre latitude is from 0  to 2pi, so in equator the colatitude will be pi?2
    angular_vel_equa= 2*np.pi*u.rad/rotation_period
    angular_velocity = angular_vel_equa*(1-relative_shear*np.sin(latitude)**2)
    return angular_velocity  

#flux function
def flux_plot(flux):
    ''' Function take a list normalizing the flux, converting the list in a csv file and rename the columns
    and return a plot    
    '''
    flux_norm = normalize([flux], norm="max")[0]
    df = pd.DataFrame(flux_norm)
    df.to_csv("lc_high.csv")
    lc = pd.read_csv("lc_high.csv")
    lc = lc.rename(columns={'Unnamed: 0': 'Days', '0': 'flux_normalized'})#-----> change the name of the columns
    lc.to_csv('lc_high_name.csv')
    return lc.plot(x="Days", y="flux_normalized", alpha=0.5)#----->simple plot of lc


# limb darkening function 
def limbdarkening(u):
    '''calculate a limb darkening lineal and put it to the spehre'''
    return (1 - u * (1 - mu))

#function that create a gaussian mask to represent spots
def spot_mask(theta_mesh, spot_theta_rad,phi_mesh,spot_phi_rad):
    '''function for creating spot mask, that could change the size and shape of the spot
    using a gaussian function
    '''
    delta_theta = np.arccos(np.cos(theta_mesh - spot_theta_rad.value))
    delta_phi = np.abs(phi_mesh - spot_phi_rad)
    distance_squared = delta_theta**2 + delta_phi**2
    sigma_squared = (spot_radius_rad / 2)**2
    return np.exp(-distance_squared / (2* sigma_squared))
    

#animation Function
#@profile
def animate(i, gray_texture, mu, spot_theta_rad, spot_phi_rad, spot_radius_rad,
            theta_mesh, phi_mesh, x, y, z, ax_sphere, elev, azim, ax_curve, line_curve, fluxes, total_frames):
    """ Function where is created the motion and call functions"""
    ax_sphere.clear()
    ax_sphere.set_axis_off()
    ax_sphere.view_init(elev=elev, azim=azim)
    texture = np.copy(gray_texture)

    spot_theta_motion = (spot_theta_rad + angular_velocity*i*cadence_time.to(u.day))

    spot_mask_motion = spot_mask(theta_mesh, spot_theta_motion, phi_mesh, spot_phi_rad)

    
    # Add texture to spot
    texture *= (1 - 0.9 * spot_mask_motion)


    # Plot surface
    #surf = ax_sphere.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=texture, shade=False)
    surf = ax_sphere.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.gray(texture), shade=False)


    total_flux = np.sum(texture)
    fluxes.append(total_flux)
    #rot.append(spot_theta_motion)
    #Normalized curve
    fluxes_normalized = normalize([fluxes], norm="max")[0]
    line_curve.set_data(np.arange(len(fluxes))*(cadence_time.to(u.day)).value, fluxes_normalized)
    #line_curve.set_data(np.arange(len(fluxes)), fluxes)

    print(f"Procesando frame {i+1}/{total_frames}, motion {spot_theta_motion:.2f}",  end='\r')  # -------> to see the process



    return[surf, line_curve]

if __name__ == '__main__':
    
    # Stellar parameters 
    r = 1* u.R_sun
    res = 60#--->resolution
    constant = 0.8  # limb darkening coefficient
    rotation_period = 1* u.day #--->rotation period of the star
    
    # Point of view of the sphere
    elev = 0
    azim = 0
    
    # parameters position of the spot
    spot_theta_deg = 0 #------>Longitude
    spot_phi_deg = 90 #------>Latitude
    spot_radius_deg = 25
    
    #parameters of how many days periods
    observing_baseline_days = 3* u.day#----> Observation time
    cadence_time = 30 * u.minute#----> Cadence
    
    # frames
    total_frames = (observing_baseline_days / cadence_time).decompose().value
    total_frames = int(total_frames) 
    
    #Parameter of differential rotation
    relative_shear_coeff = 0.2
    
    #creating angles
    theta = np.linspace(0, 2*np.pi, res)
    phi = np.linspace(0, np.pi, res)
    
    #meshgrid for creation of the sphere surface
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)
    
    x = r * np.sin(phi_mesh) * np.cos(theta_mesh)
    y = r * np.sin(phi_mesh) * np.sin(theta_mesh)
    z = r * np.cos(phi_mesh)
    
    # gray texture 
    gray_texture = np.ones((res, res))  # intensidad 1 = blanco
    
    
    #convertion paraneters
    spot_phi_rad = convertion_radians(spot_phi_deg)
    spot_radius_rad = convertion_radians(spot_radius_deg)
    spot_theta_rad = convertion_radians(spot_theta_deg)
    elev_rad = convertion_radians(elev)
    azim_rad = convertion_radians(azim)
    
    v_x = np.cos(elev_rad) * np.cos(azim_rad)
    v_y = np.cos(elev_rad) * np.sin(azim_rad)
    v_z = np.sin(elev_rad)
    
    # Limb darkening
    mu = (x * v_x + y * v_y + z * v_z) / r
    mu = np.clip(mu, 0, 1)
    
    lineal_darkening = limbdarkening(constant)
    gray_texture *= lineal_darkening
    
    #motion of spot
    angular_velocity = spot_theta(rotation_period, spot_phi_rad, relative_shear_coeff)#----->only calculate once
    
    
    
    # Figure
    plt.style.use('dark_background')# -----> dark background
    fig = plt.figure(figsize=(12, 6))
    
    ax_sphere = fig.add_subplot(121, projection='3d')
    ax_curve = fig.add_subplot(122)
    ax_sphere.set_box_aspect((0.98, 1, 0.95))#-------> symtries of the sun 
    ax_sphere.set_axis_off()
    
    ax_curve.set_title('Lightcurve', color='white')
    ax_curve.set_xlabel('Days')
    ax_curve.set_ylabel('Normalized flux')
    ax_curve.set_xlim(0, total_frames*(cadence_time.to(u.day)).value)
    ax_curve.set_ylim(0.7, 1.05)
    ax_curve.set_facecolor('black')
    line_curve, = ax_curve.plot([], [], color='orange')
    
    # Flux saves
    fluxes = []
    
    #list to add the value of radians in every frame
    #rot =[]
    
    
    ani = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=60, blit=False, repeat=False,
        fargs=(gray_texture, mu, spot_theta_rad, spot_phi_rad, spot_radius_rad,
               theta_mesh, phi_mesh, x, y, z, ax_sphere, elev, azim, ax_curve,
               line_curve, fluxes, total_frames)
    )
    ani.save('ex.gif', writer='ffmpeg', fps=20)
    plt.show()
