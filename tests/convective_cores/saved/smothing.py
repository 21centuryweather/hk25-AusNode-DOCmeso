
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



if __name__ == '__main__':
    # Create an example field (precipitation data) with random values
    np.random.seed(42)
    # Fewer points and less granular field
    lon = np.linspace(0, 360, 180)  # Fewer longitudes
    lat = np.linspace(-90, 90, 90)  # Fewer latitudes
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Random precipitation data (simulating a field)
    # Less granular precipitation data
    precip_field = np.random.rand(90, 180) * 0.5  # Reduced range (less granular)

    # Apply Gaussian smoothing
    precip_smoothed = gaussian_filter(precip_field, sigma=1)  # Adjust sigma for smoothness

    # Plot original and smoothed fields
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Plot original field
    c1 = ax[0].pcolormesh(lon_grid, lat_grid, precip_field, cmap='Blues', shading='auto')
    ax[0].set_title('Original Precipitation Field')
    fig.colorbar(c1, ax=ax[0], orientation='horizontal', label='Precipitation [mm/day]')
    ax[0].set_xlabel('Longitude')
    ax[0].set_ylabel('Latitude')

    # Plot smoothed field
    c2 = ax[1].pcolormesh(lon_grid, lat_grid, precip_smoothed, cmap='Blues', shading='auto')
    ax[1].set_title('Smoothed Precipitation Field')
    fig.colorbar(c2, ax=ax[1], orientation='horizontal', label='Precipitation [mm/day]')
    ax[1].set_xlabel('Longitude')
    ax[1].set_ylabel('Latitude')

    fig.savefig('figure.png') 







