#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
from matplotlib import cm

fig = plt.figure(facecolor = 'Black')
ax = plt.axes(projection = "3d")
ax.set_box_aspect((1, 1, 1))
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
r = 10
r_point = 10
# Convert to Cartesian coordinates
x = r*np.outer(np.cos(theta), np.sin(phi))
y = r*np.outer(np.sin(theta), np.sin(phi))
z = r*np.outer(np.ones(np.size(theta)), np.cos(phi))

#starspot

x_points = r_point * np.sin(np.pi/2) * np.cos(0)
y_points = r_point * np.sin(np.pi/2) * np.sin(0)
z_points = r_point * np.cos(np.pi/2)
def init():
    ax.plot_surface(x,y,z, cmap = 'viridis', alpha = 0.3)
    return fig

#animation
def animate(i):
    ax.view_init(elev = 2, azim = i*3)
    return fig

ax.scatter(x_points, y_points, z_points, c='k', s=50, alpha = 0.7)
ani = animation.FuncAnimation(fig, animate, init_func = init, frames = 150, interval = 200, blit =False)
plt.show()


# In[ ]:




