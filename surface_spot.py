import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation, cm
from matplotlib.colors import LightSource, Normalize

# Prepare the figure
fig = plt.figure(facecolor='Black', figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.set_box_aspect((1, 1, 1))

# Sphere parameters
r = 10
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)

# Create sphere coordinates
x = r * np.outer(np.cos(theta), np.sin(phi))
y = r * np.outer(np.sin(theta), np.sin(phi))
z = r * np.outer(np.ones(np.size(theta)), np.cos(phi))

# Stellar spot parameters
spot_lat = np.pi / 3
spot_lon = np.pi / 3
spot_radius = 0.3

# Create a mask for the stellar spot
phi_mesh, theta_mesh = np.meshgrid(phi, theta)
mask = np.exp(-((phi_mesh - spot_lat)**2 + (theta_mesh - spot_lon)**2) / (2 * spot_radius**2))
mask = mask / np.max(mask)

# Prepare light source and texture
ls = LightSource(azdeg=0, altdeg=50)
cmap = cm.get_cmap('inferno')
norm = Normalize(vmin=mask.min(), vmax=mask.max())
texture = ls.shade(mask, cmap=cmap, norm=norm, vert_exag=0.1)

# Starspot point location
x_points = r * np.sin(spot_lat) * np.cos(spot_lon)
y_points = r * np.sin(spot_lat) * np.sin(spot_lon)
z_points = r * np.cos(spot_lat)

def init():
    # Initial plot of the sphere with slight transparency
    ax.plot_surface(x, y, z, cmap='viridis', alpha=0.3)
    # Add the starspot point
    ax.scatter(x_points, y_points, z_points, c='k', s=50, alpha=0.7)
    return fig

def animate(i):
    # Clear previous view
    ax.clear()
    ax.set_box_aspect((1, 1, 1))
    
    # Rotate the view
    ax.view_init(elev=2, azim=i*3)
    
    # Replot the surface
    ax.plot_surface(x, y, z, cmap='viridis', alpha=0.3, facecolors=texture)
    
    # Replot the starspot point
    ax.scatter(x_points, y_points, z_points, c='k', s=50, alpha=0.7)
    
    # Set axis limits
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([-r, r])
    
    return fig

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=150, interval=50, blit=False)

plt.show()


