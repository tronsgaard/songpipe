import numpy as np
import matplotlib.pyplot as plt

"""
This module contains functions for plotting
"""

def plot_order_trace(flat, orders, column_range, savename=None):
    """Visualize order traced from PyReduce"""
    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca()

    vmin = np.quantile(flat, .001)
    vmax = np.quantile(flat, .85)
    ax.imshow(flat, vmin=vmin, vmax=vmax, cmap='viridis')
    ax.invert_yaxis()

    nord = len(orders)
    for i in range(nord):
        x = np.arange(*column_range[i])
        ycen = np.polyval(orders[i], x).astype(int)
        ax.plot(x, ycen, linestyle='--', color='magenta', linewidth=.5)

    # Save figure to file
    if savename is not None:
        fig.savefig(savename, dpi=150)
