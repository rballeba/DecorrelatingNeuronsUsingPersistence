import matplotlib.pyplot as plt
from seaborn import kdeplot


def paint_point_cloud_and_ppdd(persistence_diagram, point_cloud, gif_path, number, x_low=None, x_high=None,
                               y_low=None, y_high=None, axis_ppdd_low=None, axis_ppdd_high=None):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if number == 0:
        fig.suptitle(f'Initial position', fontsize=20)
    else:
        fig.suptitle(f'Iteration {number}', fontsize=20)
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1])
    if (y_low is not None) and (y_high is not None):
        ax1.set_ylim(y_low, y_high)
    if (x_low is not None) and (x_high is not None):
        ax1.set_xlim(x_low, x_high)
    if (axis_ppdd_low is not None) and (axis_ppdd_high is not None):
        ax2.set_ylim(axis_ppdd_low, axis_ppdd_high)
        ax2.set_xlim(axis_ppdd_low, axis_ppdd_high)
    ax2.plot([axis_ppdd_low, axis_ppdd_high], [axis_ppdd_low, axis_ppdd_high], 'r')
    ax2.scatter(persistence_diagram[:, 0], persistence_diagram[:, 1])
    plt.plot()
    fig.savefig(f'{gif_path}/{number}.png')
    plt.close(fig)  # It is important to close the figure


def plot_ppdd(persistence_diagram_deaths, gif_path, number, x_axis_ppdd_low=None, x_axis_ppdd_high=None,
              y_axis_ppdd_low=None, y_axis_ppdd_high=None):
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    if number == 0:
        fig.suptitle(f'Initial position', fontsize=20)
    else:
        fig.suptitle(f'Iteration {number}', fontsize=20)
    if (x_axis_ppdd_low is not None) and (x_axis_ppdd_high is not None):
        ax.set_ylim(y_axis_ppdd_low, y_axis_ppdd_high)
        ax.set_xlim(x_axis_ppdd_low, x_axis_ppdd_high)
    kdeplot(y=persistence_diagram_deaths, ax=ax)
    plt.plot()
    fig.savefig(f'{gif_path}/{number}.png')
    plt.close(fig)  # It is important to close the figure
