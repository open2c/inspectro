from functools import partial
import glob
import tempfile
import pathlib
from urllib.parse import urlparse

from cooltools.lib import numutils, runlength
from loky import get_reusable_executor
from tqdm import tqdm
import bioframe
import cooler
import numpy as np
import pandas as pd
import h5py

from utils._common import (
    split_chroms, make_chromarms, fetch_binned, assign_arms, assign_centel
)
from utils._eigdecomp import eig_trans
from utils._clustering import kmeans_sm, relabel_clusters
from utils._df2multivec import to_multivec


shell.prefix("set -euxo pipefail; ")
configfile: "config.yaml"
workdir: config['project_folder']

CHROMSIZES = bioframe.fetch_chromsizes('hg38')
CHROMOSOMES = list(CHROMSIZES[:'chrY'].index)
CONDITIONS = config["conditions"]
N_CLUSTERS = config["params"]["n_clusters"]
BINSIZE = config["params"]["binsize"]
N_EIGS = config["params"]["n_eigs"]


rule default:
    input: 
        [
            f"{{}}.{BINSIZE}.E0-E{N_EIGS}.kmeans_sm.tsv".format(condition)
            for condition in CONDITIONS
        ] + [
            f"figs/{{}}.{BINSIZE}.E0-E{N_EIGS}.kmeans_sm8.heatmap.pdf".format(condition)
            for condition in CONDITIONS
        ] + [
            f"figs/{{}}.{BINSIZE}.E0-E{N_EIGS}.kmeans_sm8.scatters.pdf".format(condition)
            for condition in CONDITIONS
        ]


rule make_bintable:
    output:
        "hg38.chromarms.{binsize}.bed",
        "hg38.bins.gc.{binsize}.pq",
    params:
        binsize=lambda wc: int(wc.binsize),
    run:
        binsize = params.binsize
        fa_records = bioframe.load_fasta(config["fasta_path"])
        centros = bioframe.fetch_centromeres('hg38').set_index('chrom')

        arms = make_chromarms(CHROMSIZES, centros['mid'], binsize)
        arms.to_csv(
            f'hg38.chromarms.{binsize}.bed', 
            sep='\t', 
            index=False, 
            header=False
        )
        armlens = (
            arms
            .assign(length=arms['end'] - arms['start'])
            .set_index('name')['length']
            .to_dict()
        )

        df = bioframe.binnify(CHROMSIZES, binsize)
        df = bioframe.frac_gc(df, fa_records)
        df = assign_arms(df, arms)
        df['armlen'] = df['arm'].apply(armlens.get)
        df['centel'] = (
            df
            .groupby('arm', sort=False)
            .apply(partial(assign_centel, arms=arms.set_index('name')))
            .reset_index(drop=True)
        )
        df['centel_abs'] = np.round(df['centel'] * df['armlen']).astype(int)
        df.to_parquet(f'hg38.bins.gc.{binsize}.pq')


rule eigdecomp:
    input:
        "hg38.bins.gc.{binsize}.pq"
    output:
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
        "figs/{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pdf",
    params:
        condition = lambda wc: wc.condition,
        binsize = lambda wc: int(wc.binsize),
        n_eigs = lambda wc: int(wc.n_eigs),
    run:
        condition = params.condition
        binsize = params.binsize
        n_eigs = params.n_eigs
        path = config["cooler_paths"][condition]
        chromosomes = list(CHROMSIZES[:'chr22'].index)  # Don't use X or Y

        # has a header (chrom, start, end, GC)
        ref_track = pd.read_parquet(
            f'hg38.bins.gc.{binsize}.pq',
        )
        ref_track = ref_track[ref_track['chrom'].isin(chromosomes)]

        # include blacklist
        if config.get("blacklist_paths") is not None:
            if condition in config["blacklist_paths"]:
                print(condition, config["blacklist_paths"][condition])

                # no header
                blacklist = pd.read_csv(
                    config["blacklist_paths"][condition],
                    sep='\t',
                    names=['chrom', 'start', 'end', 'is_bad']
                )
                ref_track = pd.merge(
                    ref_track,
                    blacklist,
                    on=['chrom', 'start', 'end'],
                    how='outer'
                )
                ref_track = ref_track[ref_track['chrom'].isin(chromosomes)]

                print(ref_track)

        clr = cooler.Cooler(path + f'::resolutions/{binsize}')

        partition = np.r_[
            [clr.offset(chrom) for chrom in chromosomes],
            clr.extent(chromosomes[-1])[1]
        ]

        eigval_df, eigvec_df = eig_trans(
            clr=clr,
            bins=ref_track,
            phasing_track_col="GC",
            n_eigs=n_eigs,
            partition=partition,
            corr_metric=None,
        )

        # Output
        eigval_df.to_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq"
        )
        eigvec_df.to_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq"
        )

        # Plot the spectrum
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'

        plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(nrows=2, ncols=1)
        plt.suptitle(f"{condition}.{binsize}")

        plt.subplot(gs[0])
        plt.stem(eigval_df['eig'][:33], eigval_df['val'][:33])

        plt.subplot(gs[1])
        sns.rugplot(eigval_df['val'])
        sns.kdeplot(eigval_df['val'], bw_adjust=0.5)
        plt.xlim(-1, 1)
        plt.savefig(f"figs/{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pdf")


rule make_multivec:
    input:
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
    output:
        "{condition}.{binsize}.E0-E{n_eigs_multivec}.trans.eigvecs.mv5"
    params:
        condition = lambda wc: wc.condition,
        binsize = lambda wc: int(wc.binsize),
        n_eigs = lambda wc: int(wc.n_eigs),
        n_eigs_multivec = lambda wc: int(wc.n_eigs_multivec)
    run:
        condition = params.condition
        binsize = params.binsize
        n_eigs = params.n_eigs
        n_eigs_multivec = params.n_eigs_multivec

        eigvals = pd.read_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq"
        )
        eigvecs = pd.read_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq"
        )

        sqrt_lam = np.sqrt(np.abs(eigvals.set_index('eig')['val'].values))
        eigvecs.iloc[:, 3:] = (
            eigvecs.iloc[:, 3:] * sqrt_lam[np.newaxis, :]
        )
        to_multivec(
            f"{condition}.{binsize}.E0-E{n_eigs_multivec}.trans.eigvecs.mv5",
            eigvecs,
            [f'E{i}' for i in range(1, n_eigs_multivec)],
            base_res=binsize,
            chromsizes=CHROMSIZES,
        )


rule clustering:
    input:
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
        "hg38.bins.gc.{binsize}.pq",
    output:
        "{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv"
    threads: 20
    params:
        condition = lambda wc: wc.condition,
        binsize = lambda wc: int(wc.binsize),
        n_eigs = lambda wc: int(wc.n_eigs),
        sort_key = "GC",
    run:
        condition = params.condition
        binsize = params.binsize
        n_eigs = params.n_eigs
        sort_key = params.sort_key

        chromosomes = list(CHROMSIZES[:'chr22'].index)  # Don't use X or Y
        keep_first = False
        weight_by_eigval = True
        positive_eigs = False

        eigvecs = pd.read_parquet(
            f'{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq'
        )
        eigvals = pd.read_parquet(
            f'{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq',
        ).set_index('eig')
        eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)]

        # Use as many eigenvectors as initial positive eigenvalues
        N_COMPONENTS = np.where(eigvals < 0)[0][0] - 1
        print(f"Using {N_COMPONENTS} components for clustering...")

        sorting_tracks = pd.read_parquet(f'hg38.bins.gc.{binsize}.pq')
        sorting_tracks = sorting_tracks[sorting_tracks['chrom'].isin(chromosomes)]

        out = eigvecs[['chrom', 'start', 'end']].copy()
 
        for n_clusters in N_CLUSTERS:

            if N_COMPONENTS is None:
                n_components = n_clusters
            else:
                n_components = N_COMPONENTS

            colname = f'kmeans_sm{n_clusters}'

            labels = kmeans_sm(
                eigvals,
                eigvecs,
                n_clusters,
                n_components,
                weight_by_eigval,
                keep_first,
                positive_eigs,
            )

            new_labels, bin_ranks = relabel_clusters(
                labels, n_clusters, sorting_tracks, sort_key
            )

            out[colname] = new_labels
            out[colname + '_order'] = bin_ranks

        if not positive_eigs:
            if keep_first:
                elo = 'E0'
                if N_COMPONENTS:
                    ehi = f'E{N_COMPONENTS - 1}'
                else:
                    ehi = 'Ek-1'
            else:
                elo = 'E1'
                if N_COMPONENTS:
                    ehi = f'E{N_COMPONENTS}'
                else:
                    ehi = 'Ek'
            which = f"{elo}-{ehi}"
        else:
            if N_COMPONENTS is None:
                which = f"positivek"
            else:
                which = f"positive{N_COMPONENTS}"

        if weight_by_eigval:
            eignorm = 'eignorm_sqrt'
        else:
            eignorm = 'eignorm_unity'

        out.to_csv(
            f'{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv',
            sep='\t',
            index=False
        )


rule agg_bigwigs:
    output:
        "tracks.{binsize}.h5"
    threads: 32
    params:
        binsize = config['params']['binsize']
    run:
        binsize = params.binsize
        ARMS = bioframe.read_table(
            f'hg38.chromarms.{binsize}.bed', schema='bed4'
        )
        ARMNAMES = ARMS['name'].tolist()
        meta = pd.read_table(config['track_metadata_path'])
        paths = meta.set_index('UID')['Path']

        bins = bioframe.binnify(CHROMSIZES, binsize)
        h5opts = dict(compression='gzip', compression_opts=6)

        with h5py.File(f'tracks.{binsize}.h5', 'w') as f:
            f.create_dataset('chrom', data=bins['chrom'].values, **h5opts)
            f.create_dataset('start', data=bins['start'].values, **h5opts)
            f.create_dataset('end', data=bins['end'].values, **h5opts)

            for ix, row in meta.iterrows():
                if row['UID'] in f:
                    continue

                if row['FileFormat'] == 'bigWig':
                    with get_reusable_executor(26) as pool:
                        acc = row['UID']
                        x = fetch_binned(
                            paths[acc], CHROMSIZES, CHROMOSOMES, BINSIZE, pool.map
                        )
                        f.create_dataset(acc, data=x, **h5opts)

                elif row['FileFormat'] == 'bedGraph':
                    acc = row['UID']
                    df = bioframe.read_table(paths[acc], schema='bedGraph')
                    ov = bioframe.overlap(
                        bins,
                        df,
                        how='left',
                        return_overlap=True,
                        keep_order=True,
                        suffixes=('', '_')
                    )
                    ov['overlap'] = ov['overlap_end'] - ov['overlap_start']
                    ov['score'] = ov['value_'] * ov['overlap']
                    out = ov.groupby(['chrom', 'start', 'end'], sort=False).agg(**{
                        'score': ('score', 'sum')
                    }).reset_index()
                    out['score'] /= (out['end'] - out['start'])
                    x = out['score'].values
                    f.create_dataset(acc, data=x, **h5opts)

                else:
                    raise ValueError(row['FileFormat'])


rule heatmap:
    input:
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
        "{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv",
        "hg38.bins.gc.{binsize}.pq",
    output:
        "figs/{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm{n_clusters}.heatmap.pdf"
    params:
        condition = lambda wc: wc.condition,
        binsize = lambda wc: int(wc.binsize),
        n_eigs = lambda wc: int(wc.n_eigs),
        n_clusters = lambda wc: int(wc.n_clusters),
    run:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'
        
        condition = params.condition
        binsize = params.binsize
        n_eigs = params.n_eigs
        n_clusters = params.n_clusters
        n_eigs_display = 10
        chromosomes = list(CHROMSIZES[:'chr22'].index)  # Don't use X or Y
        norm = 'sqrt'

        bins = pd.read_parquet(
            f'hg38.bins.gc.{binsize}.pq',
        )
        bins = bins[bins['chrom'].isin(chromosomes)]

        eigs = pd.read_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq"
        )
        eigs = eigs[eigs['chrom'].isin(chromosomes)]
        eigvals = pd.read_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq"
        ).set_index('eig')

        klust = pd.read_table(
            f"{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv",
        )
        klust = klust[klust['chrom'].isin(chromosomes)]
        klust_col = f'kmeans_sm{n_clusters}'
        
        # Ordering and coarsening
        sort_by = 'centel'
        coarse_factor = 32
        blocks = config["heatmap_groups"]
        n_tracks = sum(len(block) for block in blocks.values())
        labels = klust[klust_col].values

        if sort_by == 'centel':
            idx = np.lexsort([bins['centel_abs'].values, labels])
        # elif sort_by == 'centel-chmm':
        #     idx = np.lexsort(
        #         [bins['centel_abs'].values, klust['chromhmm'].values, labels]
        #     )
        else:
            raise ValueError(sort_by)

        lines = [run[0] for run in runlength.iterruns(labels[idx])]
        E = eigs.loc[:, 'E1':f'E{n_eigs_display}'].values
        sqrt_lam = np.sqrt(np.abs(eigvals.loc['E1':f'E{n_eigs_display}'].values))
        lo, hi = 0, lines[-1]
        extent = [-0.5, E.shape[0] - 0.5, E.shape[1] - 0.5, -0.5]

        plt.figure(figsize=(24, 8))
        gs = plt.GridSpec(
            nrows=1 + n_tracks,
            ncols=1,
            height_ratios=[6] + [1] * n_tracks,
            hspace=0,
        )

        # Eig block
        ax = ax1 = plt.subplot(gs[0])
        if norm == 'sqrt':
            X = E[idx, :].T * sqrt_lam
        else:
            X = E[idx, :].T
        X = numutils.coarsen(np.nanmean, X, {1: coarse_factor}, trim_excess=True)
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
        ax.set_yticks(np.arange(E.shape[1]))
        ax.set_xlim(0-0.5, E.shape[0]-0.5)
        ax.set_ylim(E.shape[1]-0.5, 0-0.5)
        ax.set_yticklabels([f'E{i}' for i in range(1, n_eigs_display + 1)])
        plt.vlines(lines, 0-0.5, E.shape[1]-0.5, lw=1, color='k')
        # plt.colorbar(im)
        level = 1

        # Other blocks
        options_default = {
            'cmap': 'Reds',
            'vmin': 0,
            # 'vmax': 1,
        }
        tracks = bins.copy()
        for i, name in enumerate(blocks['required'], level):
            ax = plt.subplot(gs[i], sharex=ax1)

            track = tracks[name].loc[idx]
            X = numutils.coarsen(
                np.nanmean,
                np.array([track.values]),
                {1: coarse_factor},
                trim_excess=True
            )
            # X = np.array([track.values])

            im = ax.matshow(
                X,
                rasterized=True,
                extent=extent,
                origin='lower',
                **config["tracks"][name].get("options", options_default)
            )
            ax.set_aspect('auto')
            ax.xaxis.set_visible(False)
            ax.set_xlim(lo-0.5, hi-0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([0])
            ax.set_yticklabels([name])
            plt.vlines(lines, -0.5, 0.5, lw=1, color='k')
            # plt.colorbar(im)
            level += 1

        f = h5py.File(f'tracks.{binsize}.h5', 'r')
        for block_name in blocks.keys():
            if block_name == 'required':
                continue
            for i, name in enumerate(blocks[block_name], level):
                ax = plt.subplot(gs[i], sharex=ax1)

                uid = config["tracks"][name]["uid"]
                track = f[uid][:][idx]
                X = numutils.coarsen(
                    np.nanmean,
                    np.array([track]),
                    {1: coarse_factor}, trim_excess=True
                )

                im = ax.matshow(
                    X,
                    rasterized=True,
                    extent=extent,
                    origin='lower',
                    **config["tracks"][name].get("options", options_default)
                )
                ax.set_aspect('auto')
                ax.xaxis.set_visible(False)
                ax.set_xlim(lo-0.5, hi-0.5)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([0])
                ax.set_yticklabels([name])
                plt.vlines(lines, -0.5, 0.5, lw=1, color='k')
                # plt.colorbar(im)
                level += 1

        plt.savefig(
            f"figs/{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm{n_clusters}.heatmap.pdf",
            bbox_inches='tight'
        )


rule scatters:
    input:
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        "{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
        "{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv",
    output:
        "figs/{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm{n_clusters}.scatters.pdf"
    params:
        condition = lambda wc: wc.condition,
        binsize = lambda wc: int(wc.binsize),
        n_eigs = lambda wc: int(wc.n_eigs),
        n_clusters = lambda wc: int(wc.n_clusters),
    run:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'
        from mpl_toolkits.axes_grid1 import ImageGrid
        from datashader.mpl_ext import dsshow
        import datashader as ds
        import datashader.transfer_functions as tf

        condition = params.condition
        binsize = params.binsize
        n_eigs = params.n_eigs
        n_clusters = params.n_clusters
        tracks = config["tracks"]
        chromosomes = list(CHROMSIZES[:'chr22'].index)  # Don't use X or Y

        ncols = 4
        xvar = 'E1'
        yvar = '-E2'
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

        f = h5py.File(f"tracks.{binsize}.h5", "r")

        bins = pd.read_parquet(
            f'hg38.bins.gc.{binsize}.pq',
        )
        bins = bins[bins['chrom'].isin(chromosomes)]
        eigs = pd.read_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq"
        )
        eigs = eigs[eigs['chrom'].isin(chromosomes)]
        eigs['-E2'] = -eigs['E2']
        eigvals = pd.read_parquet(
            f"{condition}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq"
        ).set_index('eig')
        klust = pd.read_table(
            f"{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv",
        )
        klust = klust[klust['chrom'].isin(chromosomes)]
        klust_col = f'kmeans_sm{n_clusters}'
        bins["cluster"] = klust[klust_col]
    
        PANELS = config["scatter_groups"]
        for panel_name, track_list in tqdm(PANELS.items()):

            print(panel_name)

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

            for i, name in enumerate(track_list, 1):
                print(name)
                ax = grid[i]
                cax = grid.cbar_axes[i]
                track = tracks[name]

                if "uid" in tracks[name]:
                    uid = tracks[name]["uid"]
                    track["data"] = f[uid][:len(eigs)]
                else:
                    track["data"] = bins[name]
                
                track_type = track.get('type', 'scalar')
                kwargs = track.get('options', {})

                if track_type == 'category':
                    kwargs['ax'] = ax
                    if 'facecolor' in kwargs:
                        ax.set_facecolor(kwargs.pop('facecolor'))
                    da = dsshow(
                        eigs.assign(cat=track['data'].astype('category')),
                        aggregator=ds.count_cat('cat'),
                        **kwargs,
                        **ds_options
                    )
                    ax.set_title(name);
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
                    kwargs['ax'] = ax
                    kwargs.setdefault('norm', 'linear')
                    kwargs.setdefault('cmap',
                        'coolwarm' if track_type == 'divergent' else 'Oranges'
                    )
                    kwargs.setdefault('vmin', 0)
                    if 'vmax' not in kwargs:
                        vopt = min(np.percentile(np.max(np.abs(track['data'])), 95), 4)
                        if track_type == 'divergent':
                            kwargs['vmin'] = -vopt
                        else:
                            kwargs['vmin'] = 0
                        kwargs['vmax'] = vopt
                    if 'facecolor' in kwargs:
                        ax.set_facecolor(kwargs.pop('facecolor'))
                    da = dsshow(
                        eigs.assign(z=track['data']),
                        aggregator=ds.mean('z'),
                        **kwargs,
                        **ds_options
                    )
                    ax.set_title(name);
                    ax.axvline(0, c='k', lw=0.5, ls='--', alpha=0.5);
                    ax.axhline(0, c='k', lw=0.5, ls='--', alpha=0.5);
                    plt.colorbar(da, cax=cax);


            plt.savefig(
                f"figs/{condition}.{binsize}.E0-E{n_eigs}.kmeans_sm{n_clusters}.scatters.pdf",
                bbox_inches='tight'
            )

