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

import utils.common as common
import utils.eigdecomp as eig

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--bins', help='parquet bins file', required=True)
parser.add_argument('--blacklist', help='blacklist file path')
parser.add_argument('--cooler', help='cooler file path', required=True)

args = parser.parse_args()

with open(args.config, "r") as infile:
    config = yaml.full_load(infile)
    
assembly = config["assembly"]
binsize = config["binsize"]
sample = config["sample"]
n_eigs = config["n_eigs"]
decomp_mode = config["decomp_mode"]

CHROMSIZES = bioframe.fetch_chromsizes(assembly)
CHROMOSOMES = list(CHROMSIZES[:'chrY'].index)
CHROMOSOMES_FOR_CLUSTERING = list(CHROMSIZES[:'chr22'].index)

try:
    CENTROMERES = bioframe.fetch_centromeres(assembly)
except ValueError:
    CENTROMERES = None

chromosomes = CHROMOSOMES_FOR_CLUSTERING

# has a header (chrom, start, end, GC)
ref_track = pd.read_parquet(args.bins)
ref_track = ref_track[ref_track['chrom'].isin(chromosomes)]

# include blacklist
if args.blacklist is not None:
    # no header
    blacklist = pd.read_csv(
        args.blacklist,
        sep='\t',
        names=['chrom', 'start', 'end']
    )
    ref_track = (
        bioframe.count_overlaps(ref_track, blacklist)
        .rename(columns={'count': 'is_bad'})
    )
ref_track = ref_track[ref_track['chrom'].isin(chromosomes)]

path = args.cooler
clr = cooler.Cooler(f"{path}::resolutions/{binsize}")

if decomp_mode=="trans":
    partition = np.r_[
        [clr.offset(chrom) for chrom in chromosomes],
        clr.extent(chromosomes[-1])[1]
    ]

    eigval_df, eigvec_df = eig.eig_trans(
        clr=clr,
        bins=ref_track,
        phasing_track_col="GC",
        n_eigs=n_eigs,
        partition=partition,
        corr_metric=None,
    )

elif decomp_mode=="cis":
    viewframe_path = (args.assembly).get("viewframe_cis", None)
    if viewframe_path is None:
        CHROMARMS = bioframe.make_chromarms(CHROMSIZES, CENTROMERES)
        viewframe = CHROMARMS.query(f"(chrom in {chromosomes})").reset_index(drop=True)
    else:
        viewframe = bioframe.load_table(viewframe_path)

    eigval_df, eigvec_df = eig.eig_cis(
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
eigval_df.to_parquet(f"{sample}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pq")
eigvec_df.to_parquet(f"{sample}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvecs.pq")

