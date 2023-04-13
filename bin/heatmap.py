#!/usr/bin/env python3
 
import matplotlib as mpl 
mpl.use("Agg")
import matplotlib.pyplot as plt 

from functools import partial
import os.path as op
import pathlib

from tqdm import tqdm
import bioframe
import cooler
import h5py
import numpy as np
import pandas as pd
import argparse
import yaml

import utils.plotting as plotting

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--eigvals', help='parquet file with eigenvalues', required=True)
parser.add_argument('--eigvecs', help='parquet file with eigenvectors', required=True)
parser.add_argument('--bins', help='parquet bins file', required=True)
parser.add_argument('--cluster', help='tsv file with k-means clusters', required=True)
parser.add_argument('--track_db', help='track_db file', required=True)
parser.add_argument('--meta', help='bigwig metadata file', required=True)

args = parser.parse_args()

with open(args.config, "r") as infile:
    config = yaml.full_load(infile)

assembly = config["assembly"]
binsize = config["binsize"]
sample = config["sample"]
n_eigs = config["n_eigs"]

CHROMSIZES = bioframe.fetch_chromsizes(assembly)
CHROMOSOMES = list(CHROMSIZES[:'chrY'].index)
CHROMOSOMES_FOR_CLUSTERING = list(CHROMSIZES[:'chr22'].index)

try:
    CENTROMERES = bioframe.fetch_centromeres(assembly)
except ValueError:
    CENTROMERES = None

chromosomes = CHROMOSOMES_FOR_CLUSTERING
sort_by = 'centel'
norm = 'sqrt'
n_eigs_heatmap = 10
n_clusters = config["n_clusters"]

eigvecs = pd.read_parquet(args.eigvecs)
eigvals = pd.read_parquet(args.eigvals).set_index('eig')['val']
sqrt_lam = np.sqrt(np.abs(eigvals.loc['E1':f'E{n_eigs_heatmap}'].to_numpy()))
if norm == 'sqrt':
    eigvecs.loc[:, 'E1':f'E{n_eigs_heatmap}'] *= sqrt_lam[np.newaxis, :]
eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)].copy()

bins = pd.read_parquet(args.bins)
clusters = pd.read_table(args.cluster)

for clus in n_clusters:
    bins["cluster"] = clusters[f'kmeans_sm{clus}']
    track_db_path = args.track_db
    if op.exists(track_db_path):
        meta = pd.read_table(args.meta).set_index("Name")
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

    plotting.plot_heatmap(
        idx,
        eigvecs.loc[:, 'E1':f'E{n_eigs_heatmap}'],
        bins,
        trackconfs= config["tracks"],
        blocks=config["heatmap_groups"],
        coarse_factor=32,
    )
    plt.savefig(f"{sample}.{binsize}.E1-E{n_eigs}.kmeansm{clus}.heatmap.pdf", bbox_inches='tight')
