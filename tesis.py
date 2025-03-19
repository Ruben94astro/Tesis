#%matplotlib notebook------------>use this in jupyter notebooko

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# time
points = 100
time = np.linspace(0, 5, points)  # ------> 5 julian days



# flux
flux = (np.sin(np.pi * time)) #-------> Modulated variations in the sinusoidal light curve


# Normalized flux
# Corrected normalized flux

# Normalized flux
if np.max(flux) != np.min(flux):  # ------------> checking if there is variation in flux
    flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
else:
    flux_norm = np.ones_like(flux)  # constant flux
    
#figures
fig, ax = plt.subplots()
radii_stellar = 8
ax.set_xlim(-radii_stellar-1, radii_stellar+1)
ax.set_ylim(-radii_stellar-1, radii_stellar+1)
ax.set_aspect('equal', 'box')

# draw star
star = plt.Circle((0, 0), radii_stellar, color='blue', ec='black', lw=2)
ax.add_artist(star)

# size of star
starspot = plt.Circle((0, 0), 0, color='red', alpha=0.7)
ax.add_artist(starspot)

# stellar area
stellar_area = np.pi * radii_stellar**2

# animation function
def motion(frame):
    # area and size spot
    area_spot = (1 - (abs(flux[frame])/np.max(flux))) * stellar_area
    size_spot = np.sqrt(area_spot / np.pi)  
    
    starspot.set_radius(size_spot)  

# run animation
ani = animation.FuncAnimation(fig, motion, frames=points, interval=100)




# Create the unnormalized light curve plot
fig2, ax2 = plt.subplots()
ax2.plot(time, flux, label="Unnormalized Flux", color="blue")
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Flux")
ax2.set_title("Unnormalized Light Curve")
ax2.legend()
ax2.grid()

# Create the normalized light curve plot
fig3, ax3 = plt.subplots()
ax3.plot(time, flux_norm, label="Normalized Flux", color="black")
ax3.set_xlabel("Time (days)")
ax3.set_ylabel("Normalized Flux")
ax3.set_title("Normalized Light Curve")
ax3.legend()
ax3.grid()

# Show the plots
plt.show()

