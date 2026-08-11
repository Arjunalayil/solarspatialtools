import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc

from solarspatialtools import spatial
from solarspatialtools.synthirrad.copula import downscale, downscale_multihour, DEFAULT_PARAMS

def matlab_compare():
    # Compares with the demo results from the original authors' MATLAB code.
    # It also demonstrates the use of the multihour helper function.
    
    tracemalloc.start()
    start_time = time.time()

    # Cloud speed and direction in radians, provided as arrays for multihour
    cs = np.array([5, 5, 5, 5, 5, 5])
    cd = np.array([0, 0, 0, 0, 0, 0]) * 2 * np.pi / 360

    # Hourly clearsky
    mean_csi = np.array([0.52, 0.71, 0.5, 0.84, 0.63, 0.22])

    # Site positions, projected to East-North coordinate plane
    lat = np.array([21.31236, 21.31303, 21.32357])
    lon = np.array([-158.08463, -158.08505, -158.08424])
    # Epos, Npos = spatial.latlon2lcs(lat, lon, lat[0], lon[0])
    Epos, Npos = spatial.lla2flat(lat, lon, lat[0], lon[0], method="tmerc")

    # The reference times
    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 00:59:59', freq='60s')

    # Multihour accepts arrays of cs, cd and mean_csi to produce multiple
    # hourly outputs at once. The results are concatenated timeseries for each
    # individual site.
    noneg = True
    scale = True
    c = downscale_multihour(times, Epos, Npos, cs, cd, mean_csi,
                            DEFAULT_PARAMS, seed=42, scale=scale, noneg=noneg)


    # Helper to plot the mean
    n_per_hour = times.shape[0]
    hcsi_block = np.repeat(mean_csi, n_per_hour)

    plt.plot(c, alpha=0.8, linewidth=1)
    plt.step(np.arange(hcsi_block.size), hcsi_block, where='post', color='k',
             linewidth=1, linestyle='--')
    plt.legend()

    # Second plot: Input CSI vs Output CSI aggregated hourly mean
    n_hours = len(mean_csi)
    n_sites = c.shape[1] if c.ndim > 1 else 1
    c_reshaped = c.values.reshape(n_hours, n_per_hour, n_sites) if hasattr(c, 'values') else c.reshape(n_hours, n_per_hour, n_sites)
    # Hourly mean per site, then average across sites
    output_hourly_mean = c_reshaped.mean(axis=1).mean(axis=1)

    hours = np.arange(1, n_hours + 1)

    plt.figure()
    plt.plot(hours, mean_csi, marker='o', linewidth=2, label='Input CSI (mean_csi)')
    plt.plot(hours, output_hourly_mean, marker='s', linewidth=2, linestyle='--', label='Output CSI (hourly mean)')
    plt.xlabel('Hour')
    plt.ylabel('CSI')
    plt.title('Input CSI vs Output Aggregated Hourly Mean CSI')
    plt.legend()
    plt.grid(True, alpha=0.3)

    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Current memory usage: {current_mem / 10**6:.4f} MB; Peak memory usage: {peak_mem / 10**6:.4f} MB")

    plt.show()


if __name__ == '__main__':
    matlab_compare()
