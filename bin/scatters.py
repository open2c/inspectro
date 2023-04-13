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
import utils.clustering as cluster

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--eigvals', help='parquet file with eigenvalues', required=True)
parser.add_argument('--eigvecs', help='parquet file with eigenvectors', required=True)
parser.add_argument('--bins', help='parquet bins file', required=True)
parser.add_argument('--clusters', help='cluster name', required=True)
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

n_clusters = config["n_clusters"]
chromosomes = CHROMOSOMES_FOR_CLUSTERING

eigvecs = pd.read_parquet(args.eigvecs)
eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)].copy()
eigvals = pd.read_parquet(args.eigvals).set_index('eig')['val']
# sqrt_lam = np.sqrt(np.abs(eigvals.to_numpy()))
# eigvecs.loc[:, 'E0':] = (
#     eigvecs.loc[:, 'E0':] * sqrt_lam[np.newaxis, :]
# )

bins = pd.read_parquet(args.bins)
clusters = pd.read_table(args.clusters)

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

    plotting.plot_scatters(
        eigvecs,
        bins,
        trackconfs=config["tracks"],
        panels=config["scatter_groups"],
    )
    plt.savefig(f"{sample}.{binsize}.E1-E{n_eigs}..kmeansm{clus}.scatter.pdf", bbox_inches='tight')
