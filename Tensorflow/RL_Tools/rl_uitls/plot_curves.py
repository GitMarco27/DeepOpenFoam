
from stable_baselines3.common.results_plotter import load_results, ts2xy

from matplotlib import pyplot as plt
import numpy as np

# Funcioni che servomo per plottare le learning curves dell'agente sb
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder,window=50,  title='Learning Curve', figsize=(10,5)):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title, figsize=figsize)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()