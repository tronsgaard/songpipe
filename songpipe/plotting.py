import numpy as np
import matplotlib.pyplot as plt

"""
This module contains functions for plotting
"""

def plot_order_trace(flat, orders, column_range, widths=None, 
                     color='magenta', savename=None, dpi=300):
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
        ax.plot(x, ycen, linestyle='--', color=color, linewidth=.5)
        if widths is not None:
            w0, w1 = widths[i]
            ax.plot(x, ycen-w0, linestyle=':', color=color, linewidth=.5)
            ax.plot(x, ycen+w1, linestyle=':', color=color, linewidth=.5)

    # Save figure to file
    if savename is not None:
        fig.savefig(savename, dpi=dpi)
