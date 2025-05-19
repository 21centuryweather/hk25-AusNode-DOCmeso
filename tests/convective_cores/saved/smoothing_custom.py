import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Function to generate exponential decay kernel
def exponential_kernel(size, decay_distance):
    """Create an exponential decay kernel of given size."""
    # Create a grid of distances from the center (distance in grid units)
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x - size//2)**2 + (y - size//2)**2)
    
    # Exponential decay
    kernel = np.exp(-dist / decay_distance)
    
    # Normalize the kernel so it sums to 1
    kernel /= kernel.sum()
    
    return kernel

# Create an example field (precipitation data) with fewer points
lon = np.linspace(0, 360, 180)  # Fewer longitudes
lat = np.linspace(-90, 90, 90)  # Fewer latitudes
precip_field = np.random.rand(90, 180) * 0.5  # Precipitation in mm/day

# Define kernel size and decay distance (80 km)
decay_distance = 80 / 111  # 80 km in degrees (roughly)

# Create the kernel
kernel_size = 7  # Size of the kernel (grid cells)
kernel = exponential_kernel(kernel_size, decay_distance)

# Apply the kernel to the precipitation field using convolution
precip_smoothed = convolve(precip_field, kernel, mode='nearest')

# Plot original and smoothed fields
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Plot original field
c1 = ax[0].pcolormesh(lon, lat, precip_field, cmap='Blues', shading='auto')
ax[0].set_title('Original Precipitation Field')
fig.colorbar(c1, ax=ax[0], orientation='horizontal', label='Precipitation [mm/day]')
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Latitude')

# Plot smoothed field
c2 = ax[1].pcolormesh(lon, lat, precip_smoothed, cmap='Blues', shading='auto')
ax[1].set_title('Smoothed Precipitation Field')
fig.colorbar(c2, ax=ax[1], orientation='horizontal', label='Precipitation [mm/day]')
ax[1].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')

fig.savefig('figure1.png') 



import numpy as np
from scipy.ndimage import minimum_filter
smoothed_field = precip_smoothed
local_minima = smoothed_field == minimum_filter(smoothed_field, size=3)
fig = plt.figure(figsize=(8, 6))
plt.pcolormesh(lon, lat, local_minima, cmap='coolwarm', shading='auto')
plt.title('Local Minima in Smoothed Field')
plt.colorbar(label='Local Minima (True/False)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
fig.savefig('figure_cores.png') 
