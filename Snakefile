import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
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

from utils.common import (
    make_chromarms, fetch_binned, assign_arms, assign_centel
)
from utils.eigdecomp import eig_trans
from utils.clustering import kmeans_sm, relabel_clusters
from utils.df2multivec import to_multivec


shell.prefix("set -euxo pipefail; ")
configfile: "config.yaml"
workdir: config['project_folder']


assembly = "hg38"
CHROMSIZES = bioframe.fetch_chromsizes(assembly)
CHROMOSOMES = list(CHROMSIZES[:'chrY'].index)
CHROMOSOMES_FOR_CLUSTERING = list(CHROMSIZES[:'chr22'].index)

CONDITIONS = config["conditions"]
binsize = config["params"]["binsize"]
n_clusters_list = config["params"]["n_clusters"]
n_eigs = config["params"]["n_eigs"]
n_eigs_multivec = 32


rule default:
    input: 
        [
            f"{{}}.{assembly}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv".format(condition)
            for condition in CONDITIONS
        ] + [
            f"figs/{{}}.{assembly}.{binsize}.E0-E{n_eigs}.kmeans_sm8.heatmap.pdf".format(condition)
            for condition in CONDITIONS
        ] + [
            f"figs/{{}}.{assembly}.{binsize}.E0-E{n_eigs}.kmeans_sm8.scatters.pdf".format(condition)
            for condition in CONDITIONS
        ]


rule make_bintable:
    output:
        chromarms = f"{assembly}.chromarms.{binsize}.bed",
        bins = f"{assembly}.bins.gc.{binsize}.pq",
    run:
        fa_records = bioframe.load_fasta(config["fasta_path"])

        try:
            centros = bioframe.fetch_centromeres(assembly)
        except ValueError:
            centros = None
        if centros is None or len(centros) == 0:
            mids = {chrom: 0 for chrom in CHROMOSOMES}
            arms = pd.DataFrame({
                "chrom": CHROMSIZES.index,
                "start": 0,
                "end": CHROMSIZES.values,
                "name": CHROMSIZES.index,
            })
        else:
            mids = centros.set_index('chrom')['mid']
            arms = make_chromarms(CHROMSIZES, mids, binsize)
        arms.to_csv(
            output.chromarms,
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
        df.to_parquet(output.bins)


rule make_track_db:
    output:
        track_db = f"tracks.{assembly}.{binsize}.h5"
    threads: 32
    run:
        h5opts = dict(compression='gzip', compression_opts=6)

        bins = bioframe.binnify(CHROMSIZES, binsize)
        if not os.exists(output.track_db):
            with h5py.File(output.track_db, 'w') as f:
                for col in [
                    'chrom',
                    'start',
                    'end',
                    'GC',
                    'armlen',
                    'centel',
                    'centel_abs'
                ]:
                    f.create_dataset(col, data=bins[col].values, **h5opts)

        meta = pd.read_table(config['track_metadata_path'])
        paths = meta.set_index('UID')['Path']
        with h5py.File(output.track_db, 'a') as f:
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


rule eigdecomp:
    input:
        bins = f"{assembly}.bins.gc.{binsize}.pq"
    output:
        eigvals = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        eigvecs = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
        eig_pdf = f"figs/{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvals.pdf",
    params:
        condition = "{condition}",
    run:
        condition = params.condition
        n_eigs_display = 32
        path = config["cooler_paths"][condition]
        chromosomes = CHROMOSOMES_FOR_CLUSTERING

        # has a header (chrom, start, end, GC)
        ref_track = pd.read_parquet(input.bins)
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
        eigval_df.to_parquet(output.eigvals)
        eigvec_df.to_parquet(output.eigvecs)

        # Plot the spectrum
        plot_spectrum(
            eigval_df,
            n_eigs_display,
            f"{condition}.{binsize}",
            output.eig_pdf
        )
        plt.savefig(output.eig_pdf)


rule make_multivec:
    input:
        eigvals = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        eigvecs = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
    output:
        multivec = f"{{condition}}.{binsize}.E0-E{n_eigs_multivec}.trans.eigvecs.mv5"
    params:
        condition = "{condition}",
    run:
        condition = params.condition

        eigvals = pd.read_parquet(input.eigvals)
        eigvecs = pd.read_parquet(input.eigvecs)

        sqrt_lam = np.sqrt(np.abs(eigvals.set_index('eig')['val'].values))
        eigvecs.loc[:, 'E0':] = (
            eigvecs.loc[:, 'E0':] * sqrt_lam[np.newaxis, :]
        )
        to_multivec(
            output.multivec,
            eigvecs,
            [f'E{i}' for i in range(1, n_eigs_multivec)],
            base_res=binsize,
            chromsizes=CHROMSIZES,
        )


rule clustering:
    input:
        bins = f"{assembly}.bins.gc.{binsize}.pq",
        eigvals = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        eigvecs = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
    output:
        clusters = f"{{condition}}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv"
    threads: 20
    params:
        sort_key = "GC",
    run:
        sort_key = params.sort_key

        chromosomes = CHROMOSOMES_FOR_CLUSTERING
        keep_first = False
        weight_by_eigval = True
        positive_eigs = False

        eigvecs = pd.read_parquet(input.eigvecs)
        eigvals = pd.read_parquet(input.eigvals).set_index('eig')
        eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)]

        # Use as many eigenvectors as initial positive eigenvalues
        N_COMPONENTS = np.where(eigvals < 0)[0][0] - 1
        print(f"Using {N_COMPONENTS} components for clustering...")

        sorting_tracks = pd.read_parquet(input.bins)
        sorting_tracks = sorting_tracks[sorting_tracks['chrom'].isin(chromosomes)]

        out = eigvecs[['chrom', 'start', 'end']].copy()
 
        for n_clusters in n_clusters_list:

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

        out.to_csv(output.clusters, sep='\t', index=False)


rule heatmap:
    input:
        eigvals = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        eigvecs = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
        clusters = f"{{condition}}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv",
        track_db = f"tracks.{assembly}.{binsize}.h5",
        bins = f"{assembly}.bins.gc.{binsize}.pq",
    output:
        heatmap_pdf = f"figs/{{condition}}.{binsize}.E0-E{n_eigs}.kmeans_sm{{n_clusters}}.heatmap.pdf"
    params:
        n_clusters = int("{n_clusters}"),
    run:
        n_clusters = params.n_clusters
        n_eigs_display = 10
        chromosomes = CHROMOSOMES_FOR_CLUSTERING
        sort_by = 'centel'
        norm = 'sqrt'

        eigs = pd.read_parquet(input.eigvecs)
        eigvals = pd.read_parquet(input.eigvals).set_index('eig')
        sqrt_lam = np.sqrt(np.abs(eigvals.loc['E1':f'E{n_eigs_display}'].values))
        if norm == 'sqrt':
            eigs.loc[:, 'E1':f'E{n_eigs_display}'] *= sqrt_lam[np.newaxis, :]
        eigs = eigs[eigs['chrom'].isin(chromosomes)].copy()

        bins = pd.read_parquet(input.bins)
        with h5py.File(input.track_db, 'r') as db:
            for group in config["heatmap_groups"].values():
                for track_name in group:
                    conf = tracks[track_name]
                    uid = conf.get("uid", track_name)
                    bins[track_name] = db[uid][:]
        bins = bins[bins['chrom'].isin(chromosomes)]
        klust = pd.read_table(input.clusters)
        klust = klust[klust['chrom'].isin(chromosomes)].copy()
        bins["cluster"] = klust[f'kmeans_sm{n_clusters}']

        if sort_by == 'centel':
            idx = np.lexsort([
                bins['centel_abs'].values, bins['cluster'].values
            ])
        else:
            raise ValueError(sort_by)

        plot_heatmap(
            idx,
            eigs['E1':f'E{n_eigs_display}'],
            bins,
            tracks=config["tracks"],
            blocks=config["heatmap_groups"],
            coarse_factor=32,
        )
        plt.savefig(output.heatmap_pdf, bbox_inches='tight')


rule scatters:
    input:
        eigvals = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvals.pq",
        eigvecs = f"{{condition}}.{binsize}.E0-E{n_eigs}.trans.eigvecs.pq",
        clusters = f"{{condition}}.{binsize}.E0-E{n_eigs}.kmeans_sm.tsv",
        track_db = f"tracks.{assembly}.{binsize}.h5",
    output:
        scatter_pdf = f"figs/{{condition}}.{binsize}.E0-E{n_eigs}.kmeans_sm{{n_clusters}}.scatters.pdf"
    params:
        n_clusters = int("{n_clusters}"),
    run:
        n_clusters = params.n_clusters
        chromosomes = CHROMOSOMES_FOR_CLUSTERING

        eigs = pd.read_parquet(input.eigvecs)
        eigs = eigs[eigs['chrom'].isin(chromosomes)].copy()
        eigvals = pd.read_parquet(input.eigvals).set_index('eig')

        bins = pd.read_parquet(input.bins)
        with h5py.File(input.track_db, 'r') as db:
            for group in config["scatter_groups"].values():
                for track_name in group:
                    conf = tracks[track_name]
                    uid = conf.get("uid", track_name)
                    bins[track_name] = db[uid][:]
        bins = bins[bins['chrom'].isin(chromosomes)]
        klust = pd.read_table(input.clusters)
        klust = klust[klust['chrom'].isin(chromosomes)].copy()
        bins["cluster"] = klust[f'kmeans_sm{n_clusters}']

        plot_scatters(
            eigs,
            bins,
            tracks=config["tracks"],
            panels=config["scatter_groups"],
        )
        plt.savefig(output.scatter_pdf, bbox_inches='tight')

