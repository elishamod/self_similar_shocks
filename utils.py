"""
useful functions for plot styling
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentError


def get_parser(doc_txt=''):
    """Usage: get_parser(__doc__)"""
    parser = argparse.ArgumentParser(description=doc_txt,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser


def smooth(x, window_len=5, window='flat'):
    """Smooth a one-dimensional array with a moving window convolution."""
    allowed_windows = ['flat', 'hanning', 'blackman', 'kaiser', 'bartlett']
    # Check input validity
    if x.ndim != 1:
        raise ValueError("smooth: x must be a 1D array.")
    if x.size < window_len:
        raise ValueError("smooth: x must be longer than window_len.")
    if window not in allowed_windows:
        raise ValueError("smooth: window must be one of: " + str(allowed_windows))
    # For window_len <= 2, no need to do anything
    if window_len < 3:
        return x
    # window_len must be odd - decrease it if even
    if window_len % 2 == 0:
        window_len -= 1
    # Create filter
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval(f'np.{window}(window_len)')
    # Convlove padded vector with filter
    half_len = (window_len - 1) // 2
    s = np.r_[x[half_len:0:-1], x, x[-2:-half_len - 2:-1]]
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def finish_styled_plot(grid_minor=True):
    """
    This function is meant to be used after plotting using init_styled_plot.
    It sets the figure to be in tight layout, fix the legend if necessary and the grid minor.
    :param grid_minor: choose if to plot minor grid lines
    """
    for i in plt.get_fignums():
        fig = plt.figure(i)
        fig.tight_layout()
        for ax in fig.axes:
            labels, _ = ax.get_legend_handles_labels()
            if len(labels) > 0:
                curr_legend = ax.get_legend()
                if curr_legend is None:
                    curr_legend = ax.legend()
                curr_legend.set_draggable(True)
            ax.grid(visible=True, which='major', color='#d7d7d7', linestyle='-')
            if grid_minor:
                ax.grid(visible=True, which='minor', color='#d9d9d9', linestyle=':')
            else:
                ax.grid(visible=False, which='minor')


CB_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#965628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def CBC(CB):
    CB = np.mod(CB, len(CB_colors)).astype(int)
    return np.take(CB_colors, CB)


def init_styled_plot(tex=False, cmap='tab20', font_type='David'):
    import warnings
    from matplotlib.mathtext import MathTextWarning
    warnings.filterwarnings('ignore', category=MathTextWarning)
    for attr, val in styled_plots(tex=tex, cmap=cmap, font_type=font_type).items():
        plt.rcParams[attr] = val
    return


def styled_plots(tex=False, cmap='tab20', font_type='David'):
    """
    This function returns a dictionary of rcParams to change the plot to look like matlab's.
    This function works better with finish_styled_plots().
    :param tex: using mathematical symbols
    :param cmap: color scheme for lines
    :param font_type: font to use in titles and labels
    """
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' \
                                          r'\usepackage{amssymb}' \
                                          r'\usepackage{bm}' \
                                          r'\usepackage[T1]{fontenc}'
    plt.rcParams['font.family'] = font_type
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.get_cmap(cmap).colors)

    return {
        'text.usetex': tex,
        'font.family': font_type,
        'font.serif': font_type,

        'mathtext.fontset': 'custom',
        'mathtext.cal': font_type,
        'mathtext.bf': font_type,
        'mathtext.it': font_type,
        'mathtext.rm': font_type,
        'mathtext.sf': font_type,
        'mathtext.tt': font_type,
        'axes.unicode_minus': False,

        'legend.framealpha': 1,
        'legend.edgecolor': '0',
        'legend.fancybox': False,
        'legend.title_fontsize': 18,
        'legend.fontsize': 16,
        'legend.loc': 'best',

        'patch.linewidth': 0.7,
        'lines.linewidth': 3,
        'lines.markersize': 6.0,

        'font.size': 24,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'axes.linewidth': 0.7,
        'axes.grid.which': 'both',
        'axes.grid': True,
        'grid.alpha': 1,
        'grid.linewidth': 1,
        'grid.linestyle': '-',

        'figure.figsize': (8, 6),

        'errorbar.capsize': 3,

        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,

        'savefig.dpi': 500,
        'savefig.format': 'svg',

        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.13,
        'figure.subplot.right': 0.95,
        'figure.subplot.top': 0.9,
    }
