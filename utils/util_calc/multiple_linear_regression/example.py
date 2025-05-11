'''
# ----------------------
#   util_calc example
# ----------------------

'''

# == imports ==
# -- packages --
import numpy as np
import matplotlib.pyplot as plt

# -- imported scripts --
import os
import sys
import importlib
sys.path.insert(0, os.getcwd())
import utils.util_calc.multiple_linear_regression.mlr_calc as mlR



def save_fig(fig, folder, filename):
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    fig.savefig(path)
    print(f'plot saved at: {path}')
    plt.close(fig)

def plot(y, x_list):
    folder = f'{os.path.dirname(__file__)}/plots'
    filename = f'mlr_plot.png'
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter1 = ax.scatter(x_list[0], x_list[1], c=y, cmap='viridis')
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('y values')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    save_fig(fig, folder, filename)




# == when this script is ran ==
if __name__ == '__main__':
    # -- response variable --
    y = np.array([1, 2, 3, 4, 5])

    # -- "independent" variables --
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([5, 4, 3, 2, 1]) 
    x_list = [mlR.standardize_variable(x1),  mlR.standardize_variable(x2)]


    # -- plot --
    plot_it = True
    if plot_it:
        plot(y, x_list)
    exit()

    # -- calculate contribution to correlation --
    mlR.calculate_mlr(y, x_list)








