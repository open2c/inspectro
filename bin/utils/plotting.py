#!/usr/bin/env python3

import numpy as np
import pandas as pd
from cooltools.lib import numutils, runlength
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from datashader.mpl_ext import dsshow
import datashader as ds
import datashader.transfer_functions as tf
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def plot_spectrum(eigval_df, n_eigs_display, title, outpath):
    fig = plt.figure(figsize=(15, 8))
    gs = plt.GridSpec(nrows=2, ncols=1)
    plt.suptitle(title)
    plt.subplot(gs[0])
    plt.stem(
        eigval_df['eig'][:n_eigs_display + 1],
        eigval_df['val'][:n_eigs_display + 1]
    )
    plt.subplot(gs[1])
    sns.rugplot(eigval_df['val'])
    sns.kdeplot(eigval_df['val'], bw_adjust=0.5)
    plt.xlim(-1, 1)
    return fig


def plot_heatmap(
    idx,
    eigs,
    bins,
    trackconfs,
    blocks,
    coarse_factor=32,
    options_default=None
):
    if options_default is None:
        options_default = {
            'cmap': 'Reds',
            'vmin': 0,
        }
    E = eigs.values[idx, :].T
    labels = bins["cluster"].values
    lines = [run[0] for run in runlength.iterruns(labels[idx])]
    lo, hi = 0, lines[-1]
    extent = [-0.5, E.shape[1] - 0.5, E.shape[0] - 0.5, -0.5]

    n_tracks = sum(len(block) for block in blocks.values())
    fig = plt.figure(figsize=(24, 8))
    gs = plt.GridSpec(
        nrows=1 + n_tracks,
        ncols=1,
        height_ratios=[6] + [1] * n_tracks,
        hspace=0,
    )

    # Eig block
    ax = ax1 = plt.subplot(gs[0])
    X = numutils.coarsen(
        np.nanmean,
        E,
        {1: coarse_factor},
        trim_excess=True
    )
    ax.matshow(
        X,
        rasterized=True,
        extent=extent,
        cmap='RdBu_r',
        vmin=-0.005,
        vmax=0.005,
    )
    ax.set_aspect('auto')
    ax.xaxis.set_visible(False)
    ax.set_yticks(np.arange(E.shape[0]))
    ax.set_xlim(-0.5, E.shape[1] - 0.5)
    ax.set_ylim(E.shape[0] - 0.5, -0.5)
    ax.set_yticklabels([f'E{i}' for i in range(1, E.shape[0] + 1)])
    plt.vlines(lines, -0.5, E.shape[1]-0.5, lw=1, color='k')
    # plt.colorbar(im)
    level = 1

    # Other blocks
    for block_name in blocks.keys():
        for i, track_name in enumerate(blocks[block_name], level):
            track_conf = trackconfs[track_name]
            ax = plt.subplot(gs[i], sharex=ax1)
            x = bins[track_name].values[idx]

            track_conf = trackconfs[track_name]
            track_type = track_conf.get('type', 'scalar')
            kwargs = track_conf.get('options', {})

            kwargs.setdefault('cmap',
                'RdBu_r' if track_type == 'divergent' else 'Reds'
            )
            kwargs.setdefault('vmin', 0)
            if 'vmax' not in kwargs:
                vopt = np.percentile(np.max(np.abs(x)), 90)
                if track_type == 'divergent':
                    kwargs['vmin'] = -vopt
                else:
                    kwargs['vmin'] = 0
                kwargs['vmax'] = vopt

            X = numutils.coarsen(
                np.nanmean,
                np.array([x]),
                {1: coarse_factor},
                trim_excess=True
            )
            im = ax.matshow(
                X,
                rasterized=True,
                extent=extent,
                origin='lower',
                **kwargs
            )
            ax.set_aspect('auto')
            ax.xaxis.set_visible(False)
            ax.set_xlim(lo - 0.5, hi - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([0])
            ax.set_yticklabels([track_name])
            plt.vlines(lines, -0.5, 0.5, lw=1, color='k')
            # plt.colorbar(im)
            level += 1
    return fig


def plot_scatters(eigs, bins, trackconfs, panels):
    ncols = 4
    xvar = 'E1'
    yvar = 'E2'
    lo, hi = -0.015, 0.015
    tick_lo, tick_hi = -0.01, 0.01
    ds_options = {
        'glyph': ds.Point(xvar, yvar),
        'x_range': [lo, hi],
        'y_range': [lo, hi],
        # 'shade_hook': tf.dynspread, #partial(tf.dynspread, threshold=0.75, max_px=4),
        'aspect': 'equal',
        'rasterized': True,
        'interpolation': 'none'
    }

    for panel_name, track_list in panels.items():

        nrows = int(np.ceil((len(track_list) + 1)/ncols))
        gridshape = (nrows, ncols)
        fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
        grid = ImageGrid(
            fig, 111, gridshape,
            share_all=True,
            cbar_mode="each",
            cbar_pad=0.05,
            axes_pad=0.5
        )
        grid[0].set_xticks([tick_lo, 0, tick_hi])
        grid[0].set_yticks([tick_lo, 0, tick_hi])
        for i in range(gridshape[0]):
            grid[i*gridshape[1]].set_ylabel(yvar)
        for j in range(gridshape[1]):
            grid[gridshape[1]*(gridshape[0] - 1) + j].set_xlabel(xvar)

        # Density
        ax = grid[0]
        cax = grid.cbar_axes[0]
        da = dsshow(
            eigs,
            norm='linear',
            cmap='viridis',
            ax=ax,
            **ds_options
        )
        ax.set_title('density')
        ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.5)
        ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.5)
        plt.colorbar(da, cax=cax)

        # Other signals
        for i, track_name in enumerate(track_list, 1):
            ax = grid[i]
            cax = grid.cbar_axes[i]

            track_conf = trackconfs[track_name]
            track_type = track_conf.get('type', 'scalar')
            kwargs = track_conf.get('options', {})

            if track_type == 'category':
                df = eigs.assign(
                    z=bins[track_name].astype('category')
                )
                kwargs['ax'] = ax
                if 'facecolor' in kwargs:
                    ax.set_facecolor(kwargs.pop('facecolor'))
                da = dsshow(
                    df,
                    aggregator=ds.count_cat('z'),
                    **kwargs,
                    **ds_options
                )
                ax.set_title(track_name);
                ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.5);
                ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.5);
                cax.legend(
                    handles=da.get_legend_elements(),
                    fontsize=6,
                    borderaxespad=0,
                    loc='upper left'
                );
                cax.axis("off");
            else:
                df = eigs.assign(z=bins[track_name])
                kwargs['ax'] = ax
                kwargs.setdefault('norm', 'linear')
                kwargs.setdefault('cmap',
                    'RdBu_r' if track_type == 'divergent' else 'Oranges'
                )
                kwargs.setdefault('vmin', 0)
                if 'vmax' not in kwargs:
                    vopt = np.percentile(np.max(np.abs(df['z'])), 90)
                    if track_type == 'divergent':
                        kwargs['vmin'] = -vopt
                    else:
                        kwargs['vmin'] = 0
                    kwargs['vmax'] = vopt
                if 'facecolor' in kwargs:
                    ax.set_facecolor(kwargs.pop('facecolor'))
                da = dsshow(
                    df,
                    aggregator=ds.mean('z'),
                    **kwargs,
                    **ds_options
                )
                ax.set_title(track_name);
                ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.5);
                ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.5);
                plt.colorbar(da, cax=cax);

                