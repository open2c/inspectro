import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from functools import partial
import os.path as op
import pathlib

from loky import get_reusable_executor
from tqdm import tqdm
import bioframe
import cooler
import h5py
import numpy as np
import pandas as pd

from inspectro import utils
from inspectro.utils.common import (
    make_chromarms, fetch_binned, assign_arms, assign_centel
)
from inspectro.utils.eigdecomp import eig_trans, eig_cis
from inspectro.utils.clustering import kmeans_sm, relabel_clusters
from inspectro.utils.df2multivec import to_multivec
from inspectro.utils.plotting import plot_spectrum, plot_heatmap, plot_scatters


shell.prefix("set -euxo pipefail; ")
configfile: "config.yaml"
workdir: "results/"


assembly = config["assembly"]["name"]
CHROMSIZES = bioframe.fetch_chromsizes(assembly)
CHROMOSOMES = list(CHROMSIZES[:'chrY'].index)
CHROMOSOMES_FOR_CLUSTERING = list(CHROMSIZES[:'chr22'].index)

try:
    CENTROMERES = bioframe.fetch_centromeres(assembly)
except ValueError:
    CENTROMERES = None

samples = list(config["samples"].keys())
binsize = config["params"]["binsize"]
n_clusters_list = config["params"]["n_clusters"]
n_eigs = config["params"]["n_eigs"]
n_eigs_multivec = 32
n_eigs_heatmap = 10
decomp_mode = config["params"]["decomp_mode"]


def generate_targets(wc):

    targets = []
    for sample in samples:
        # Do not write clusters and plots for cis
        # if decomp_mode == "cis":
        targets.append(
            f"{sample}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq"
        )
        targets.append(
            f"{sample}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvecs.pq"
        )
        # else:
        #     targets.append(
        #         f"{sample}.{binsize}.E1-E{n_eigs}.kmeans_sm.tsv"
        #     )
        #     targets.append(
        #         f"figs/{sample}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pdf"
        #     )
        #     targets.extend(expand(
        #         f"figs/{sample}.{binsize}.E1-E{n_eigs}.kmeans_sm{{n}}.heatmap.pdf",
        #         n=n_clusters_list,
        #     ))
        #     targets.extend(expand(
        #         f"figs/{sample}.{binsize}.E1-E{n_eigs}.kmeans_sm{{n}}.scatters.pdf",
        #         n=[n for n in n_clusters_list if n < 20],
        #     ))
    return targets


rule default:
    input: generate_targets


rule make_bintable:
    output:
        chromarms = f"{assembly}.chromarms.{binsize}.bed",
        bins = f"{assembly}.bins.gc.{binsize}.pq",
    run:
        if CENTROMERES is None or len(CENTROMERES) == 0:
            mids = {chrom: 0 for chrom in CHROMOSOMES}
            arms = pd.DataFrame({
                "chrom": CHROMSIZES.index,
                "start": 0,
                "end": CHROMSIZES.values,
                "name": CHROMSIZES.index,
            })
        else:
            mids = CENTROMERES.set_index('chrom')['mid']
            arms = make_chromarms(CHROMSIZES, mids, binsize)
        arms.to_csv(
            output.chromarms,
            sep='\t',
            index=False,
            header=False
        )

        fa_records = bioframe.load_fasta(config["assembly"]["fasta_path"])
        df = bioframe.binnify(CHROMSIZES, binsize)
        df = bioframe.frac_gc(df, fa_records)
        df = assign_arms(df, arms)
        armlens = (
            arms
            .assign(length=arms['end'] - arms['start'])
            .set_index('name')['length']
            .to_dict()
        )
        df['armlen'] = df['arm'].apply(armlens.get)
        df['centel'] = (
            df
            .groupby('arm', sort=False)
            .apply(partial(assign_centel, arms=arms.set_index('name')))
            .reset_index(drop=True)
        )
        df['centel_abs'] = np.round(df['centel'] * df['armlen']).astype(int)
        df.to_parquet(output.bins)


rule eigdecomp:
    input:
        bins = f"{assembly}.bins.gc.{binsize}.pq"
    output:
        eigvals = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq",
        eigvecs = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvecs.pq",
    threads: workflow.cores
    params:
        sample = "{sample}",
    run:
        sample = params.sample
        chromosomes = CHROMOSOMES_FOR_CLUSTERING

        # has a header (chrom, start, end, GC)
        ref_track = pd.read_parquet(input.bins)
        ref_track = ref_track[ref_track['chrom'].isin(chromosomes)]

        # include blacklist
        if config["samples"][sample].get("blacklist_path") is not None:
            # no header
            blacklist = pd.read_csv(
                config["samples"][sample]["blacklist_path"],
                sep='\t',
                names=['chrom', 'start', 'end']
            )
            ref_track = (
                bioframe.count_overlaps(ref_track, blacklist)
                .rename(columns={'count': 'is_bad'})
            )
        ref_track = ref_track[ref_track['chrom'].isin(chromosomes)]

        path = config["samples"][sample]["cooler_path"]
        clr = cooler.Cooler(f'{path}::resolutions/{binsize}')

        if decomp_mode=="trans":
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

        elif decomp_mode=="cis":
            viewframe_path = config["assembly"].get("viewframe_cis", None)
            if viewframe_path is None:
                CHROMARMS = bioframe.make_chromarms(CHROMSIZES, CENTROMERES)
                viewframe = CHROMARMS.query(f"(chrom in {chromosomes})").reset_index(drop=True)
            else:
                viewframe = bioframe.load_table(viewframe_path)

            eigval_df, eigvec_df = eig_cis(
                clr=clr,
                bins=ref_track,
                phasing_track_col="GC",
                n_eigs=n_eigs,
                corr_metric=None,
                ignore_diags=None, # will be inferred from cooler
                view_df=viewframe
            )
        else:
            raise ValueError(f"Mode {decomp_mode} is not implemented")

        # Output
        eigval_df.to_parquet(output.eigvals)
        eigvec_df.to_parquet(output.eigvecs)


rule plot_spectrum:
    input:
        eigvals = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq"
    output:
        eig_pdf = f"figs/{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pdf"
    params:
        sample = "{sample}"
    run:
        # Plot the spectrum
        eigval_df = pd.read_parquet(input.eigvals)
        plot_spectrum(
            eigval_df,
            n_eigs_display=min(32, n_eigs),
            title=f"{params.sample}.{binsize}",
            outpath=output.eig_pdf
        )
        plt.savefig(output.eig_pdf)


rule make_multivec:
    input:
        eigvals = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq",
        eigvecs = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvecs.pq",
    output:
        multivec = f"{{sample}}.{binsize}.E0-E{n_eigs_multivec}.{decomp_mode}.eigvecs.mv5"
    run:
        eigvals = pd.read_parquet(input.eigvals).set_index('eig')['val']
        eigvecs = pd.read_parquet(input.eigvecs)

        sqrt_lam = np.sqrt(np.abs(eigvals.to_numpy()))
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
        eigvals = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq",
        eigvecs = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvecs.pq",
    output:
        clusters = f"{{sample}}.{binsize}.E1-E{n_eigs}.kmeans_sm.tsv"
    threads: 32
    run:
        chromosomes = CHROMOSOMES_FOR_CLUSTERING
        keep_first = False
        weight_by_eigval = True
        positive_eigs = False
        cluster_sort_key = "GC"

        eigvecs = pd.read_parquet(input.eigvecs)
        eigvals = pd.read_parquet(input.eigvals).set_index('eig')
        eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)]

        # Use as many eigenvectors as initial positive eigenvalues
        n_components = np.where(eigvals < 0)[0][0] - 1
        print(f"Using {n_components} components for clustering...")

        sorting_tracks = pd.read_parquet(input.bins)
        sorting_tracks = sorting_tracks[sorting_tracks['chrom'].isin(chromosomes)]

        out = eigvecs[['chrom', 'start', 'end']].copy()

        for n_clusters in n_clusters_list:

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
                labels, n_clusters, sorting_tracks, cluster_sort_key
            )

            out[colname] = new_labels
            out[colname + '_order'] = bin_ranks

        out.to_csv(output.clusters, sep='\t', index=False)


rule make_track_db:
    input:
        bins = f"{assembly}.bins.gc.{binsize}.pq"
    output:
        track_db = f"tracks.{assembly}.{binsize}.h5"
    threads: 32
    run:
        h5opts = dict(compression='gzip', compression_opts=6)

        bins = pd.read_parquet(input.bins)
        if not op.exists(output.track_db):
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

        meta = pd.read_table(config['bigwig_metadata_path'])
        paths = meta.set_index('ID')['Path']
        with h5py.File(output.track_db, 'a') as f:
            for ix, row in meta.iterrows():
                if row['ID'] in f:
                    continue

                if row['FileFormat'].lower() == 'bigwig':
                    with get_reusable_executor(26) as pool:
                        acc = row['ID']
                        x = fetch_binned(
                            paths[acc],
                            CHROMSIZES,
                            CHROMOSOMES,
                            binsize,
                            pool.map
                        )
                        f.create_dataset(acc, data=x, **h5opts)

                elif row['FileFormat'].lower() == 'bedgraph':
                    acc = row['ID']
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
        eigvals = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq",
        eigvecs = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvecs.pq",
        clusters = f"{{sample}}.{binsize}.E1-E{n_eigs}.kmeans_sm.tsv",
        # track_db = f"tracks.{assembly}.{binsize}.h5",
        bins = f"{assembly}.bins.gc.{binsize}.pq",
    output:
        heatmap_pdf = f"figs/{{sample}}.{binsize}.E1-E{n_eigs}.kmeans_sm{{n_clusters}}.heatmap.pdf"
    params:
        n_clusters = lambda wc: int(wc.n_clusters),
    run:
        n_clusters = params.n_clusters
        chromosomes = CHROMOSOMES_FOR_CLUSTERING
        sort_by = 'centel'
        norm = 'sqrt'

        eigvecs = pd.read_parquet(input.eigvecs)
        eigvals = pd.read_parquet(input.eigvals).set_index('eig')['val']
        sqrt_lam = np.sqrt(np.abs(eigvals.loc['E1':f'E{n_eigs_heatmap}'].to_numpy()))
        if norm == 'sqrt':
            eigvecs.loc[:, 'E1':f'E{n_eigs_heatmap}'] *= sqrt_lam[np.newaxis, :]
        eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)].copy()

        bins = pd.read_parquet(input.bins)
        clusters = pd.read_table(input.clusters)
        bins["cluster"] = clusters[f'kmeans_sm{n_clusters}']
        track_db_path = f"tracks.{assembly}.{binsize}.h5"
        if op.exists(track_db_path):
            meta = pd.read_table(config['bigwig_metadata_path']).set_index("Name")
            with h5py.File(track_db_path, 'r') as db:
                for group in config["scatter_groups"].values():
                    for track_name in group:
                        if track_name not in bins.columns:
                            uid = meta["ID"].get(track_name, track_name)
                            bins[track_name] = db[uid][:]
        bins = bins[bins['chrom'].isin(chromosomes)].copy()

        if sort_by == 'centel':
            idx = np.lexsort([
                bins['centel_abs'].values, bins['cluster'].values
            ])
        else:
            raise ValueError(sort_by)

        plot_heatmap(
            idx,
            eigvecs.loc[:, 'E1':f'E{n_eigs_heatmap}'],
            bins,
            trackconfs=config["tracks"],
            blocks=config["heatmap_groups"],
            coarse_factor=32,
        )
        plt.savefig(output.heatmap_pdf, bbox_inches='tight')


rule scatters:
    input:
        eigvals = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq",
        eigvecs = f"{{sample}}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvecs.pq",
        clusters = f"{{sample}}.{binsize}.E1-E{n_eigs}.kmeans_sm.tsv",
        # track_db = f"tracks.{assembly}.{binsize}.h5",
        bins = f"{assembly}.bins.gc.{binsize}.pq",
    output:
        scatter_pdf = f"figs/{{sample}}.{binsize}.E1-E{n_eigs}.kmeans_sm{{n_clusters}}.scatters.pdf"
    params:
        n_clusters = lambda wc: wc.n_clusters
    run:
        n_clusters = params.n_clusters
        chromosomes = CHROMOSOMES_FOR_CLUSTERING

        eigvecs = pd.read_parquet(input.eigvecs)
        eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)].copy()
        eigvals = pd.read_parquet(input.eigvals).set_index('eig')['val']
        # sqrt_lam = np.sqrt(np.abs(eigvals.to_numpy()))
        # eigvecs.loc[:, 'E0':] = (
        #     eigvecs.loc[:, 'E0':] * sqrt_lam[np.newaxis, :]
        # )

        bins = pd.read_parquet(input.bins)
        clusters = pd.read_table(input.clusters)
        bins["cluster"] = clusters[f'kmeans_sm{n_clusters}']
        track_db_path = f"tracks.{assembly}.{binsize}.h5"
        if op.exists(track_db_path):
            meta = pd.read_table(config['bigwig_metadata_path']).set_index("Name")
            with h5py.File(track_db_path, 'r') as db:
                for group in config["scatter_groups"].values():
                    for track_name in group:
                        if track_name not in bins.columns:
                            uid = meta["ID"].get(track_name, track_name)
                            bins[track_name] = db[uid][:]
        bins = bins[bins['chrom'].isin(chromosomes)].copy()

        plot_scatters(
            eigvecs,
            bins,
            trackconfs=config["tracks"],
            panels=config["scatter_groups"],
        )
        plt.savefig(output.scatter_pdf, bbox_inches='tight')
