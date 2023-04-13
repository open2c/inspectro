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

import utils.clustering as cluster

parser = argparse.ArgumentParser()

parser.add_argument('--config')
parser.add_argument('--eigvals', help='parquet file with eigenvalues', required=True)
parser.add_argument('--eigvecs', help='parquet file with eigenvectors', required=True)
parser.add_argument('--bins', help='parquet bins file', required=True)

args = parser.parse_args()

with open(args.config, "r") as infile:
    config = yaml.full_load(infile)
    
assembly = config["assembly"]
n_clusters = config["n_clusters"]
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
keep_first = False
weight_by_eigval = True
positive_eigs = False
cluster_sort_key = "GC"

eigvecs = pd.read_parquet(args.eigvecs)
eigvals = pd.read_parquet(args.eigvals).set_index('eig')
eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)]

# Use as many eigenvectors as initial positive eigenvalues
n_components = np.where(eigvals < 0)[0][0] - 1
print(f"Using {n_components} components for clustering...")

sorting_tracks = pd.read_parquet(args.bins)
sorting_tracks = sorting_tracks[sorting_tracks['chrom'].isin(chromosomes)]

out = eigvecs[['chrom', 'start', 'end']].copy()

for n_cluster in n_clusters:

    colname = f'kmeans_sm{n_cluster}'

    labels = cluster.kmeans_sm(
        eigvals,
        eigvecs,
        n_cluster,
        n_components,
        weight_by_eigval,
        keep_first,
        positive_eigs,
    )

    new_labels, bin_ranks = cluster.relabel_clusters(
        labels, n_cluster, sorting_tracks, cluster_sort_key
    )

    out[colname] = new_labels
    out[colname + '_order'] = bin_ranks

out.to_csv(f"{sample}.{binsize}.E1-E{n_eigs}.kmeans_sm.tsv", sep='\t', index=False)

