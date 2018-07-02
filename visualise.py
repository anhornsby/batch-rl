# coding: utf-8
"""
Batch Reinforcement Learning plots
Adam Hornsby
"""

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_fqi_results(mean_timesteps, timesteps_baseline, losses, save_path):
    """Create necessary plots following an FQI batch"""

    # plot the number of time steps
    plot_timesteps(mean_timesteps, timesteps_baseline, save_path)

    #  plot bellman loss for each iteration
    plot_losses(losses, save_path)


def plot_losses(losses, save_path=None):
    """Plot Bellman loss for each training iteration"""

    plt.plot(np.arange(len(losses)), losses, label='Bellman loss');

    plt.ylabel('Bellman Loss');
    plt.xlabel('Training iteration');

    if save_path is not None:
        plt.tight_layout();
        plt.savefig(save_path + '/bellman_loss.png');

    plt.clf();
    plt.cla();


def plot_timesteps(mean_timesteps, timesteps_baseline, save_path=None):
    """Plot the mean number of timesteps reached for each training iteration"""

    # create the plot
    plt.plot(np.arange(len(mean_timesteps)) + 1, mean_timesteps, label='Agent');

    plt.plot(np.arange(len(mean_timesteps)) + 1, [timesteps_baseline] * len(mean_timesteps),
             linestyle='--',
             color='green',
             label='Random baseline');

    plt.xlabel('Training iteration');
    plt.ylabel('Mean timesteps (#)');

    plt.legend();

    if save_path is not None:
        plt.tight_layout();
        plt.savefig(save_path + '/n_timesteps.png');

    plt.clf();
    plt.cla();
