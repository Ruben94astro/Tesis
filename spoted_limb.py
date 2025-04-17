import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Sphere parameters
r = 10 ##----> radii
res = 70 #-----> resolution "how many squares are forming my sphere"

# Making a grid
theta = np.linspace(0, 2*np.pi, res) #------> 0 to 2pi because is longitude 
phi = np.linspace(0, np.pi, res) #------> 0 to pi because is latitude
theta_mesh, phi_mesh = np.meshgrid(theta, phi) #--------> mesh grid

# Transform spherical coordinates to cartesian
x = r * np.sin(phi_mesh) * np.cos(theta_mesh)
y = r * np.sin(phi_mesh) * np.sin(theta_mesh)
z = r * np.cos(phi_mesh)

# color matrix
base_texture = np.ones((res, res, 4)) #-----> represent the a matrix with RGB color and opacity
base_texture[:, :, 0:3] = [1.0, 0.65, 0] #-------> creating orange color
base_texture[:, :, 3] = 1 #---------> opacity

# Coordinates of spot
spot_theta_deg = 0
spot_phi_deg   = 90
spot_radius_deg = 10

#convert degrees in position from zero to resolution
spot_theta_idx = int((spot_theta_deg / 360) * res)
spot_phi_idx   = int((spot_phi_deg   / 180) * res)
spot_radius_pix = int((spot_radius_deg / 180) * res)

# Coeficient of limb darkening
u = 0.7

# limb darkening linear
mu = y / r #--------> we took the axe y because is our point of view
mu = np.clip(mu, 0, 1)  #------>take only integer values

# limb darkening for every channel
for c in range(3):  # R, G, B
    base_texture[:,:, c] *= (1 - u * (1 - mu))#----> i dont know but works!!!


# plotting 3d figures
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1,1,1))
ax.view_init(elev=0, azim=95)
ax.set_axis_off()

#animation
def animate(i):
    ax.clear()  # ------->clear every frame
    ax.set_axis_off()
    ax.set_box_aspect((1,1,1))#------->putting simetry to our sphere

    # Reebot texture
    texture = np.copy(base_texture)

    # new position of spot
    Y, X = np.ogrid[:res, :res]
    distance_squared = ((X - (spot_theta_idx + i*2) % res)**2 + (Y/2 - spot_phi_idx/2)**2)#---->cartesian distance
    sigma_squared = (spot_radius_pix / 2)**2
    spot_mask = np.exp(-distance_squared / (2 * sigma_squared))

    # putting color on the mask spot
    texture[:, :, 0:3] *= (1 - 0.9 * spot_mask[:, :, np.newaxis])

        # new position of spot #2
    distance_squared2 = ((X - (spot_theta_idx +i) % res)**2 + (Y - (spot_phi_idx-20))**2)#---->cartesian distance
    sigma_squared2 = ((spot_radius_pix)/ 2)**2
    spot_mask2 = np.exp(-distance_squared2 / (2 * sigma_squared2))

    # putting color on the mask spot
    texture[:, :, 0:3] *= (1 - 0.9 * spot_mask[:, :, np.newaxis])
    texture[:, :, 0:3] *= (1 - 0.9 * spot_mask2[:, :, np.newaxis])
    


    # create a new surface
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=texture, shade=False)

    return [surf]

ani = animation.FuncAnimation(fig, animate, frames=120, interval=50, blit=False)
plt.show()
