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

import utils.df2multivec as mv

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--eigvals', help='parquet file with eigenvalues', required=True)
parser.add_argument('--eigvecs', help='parquet file with eigenvectors', required=True)

args = parser.parse_args()

with open(args.config, "r") as infile:
    config = yaml.full_load(infile)
    
assembly = config["assembly"]
binsize = config["binsize"]
sample = config["sample"]
n_eigs_multivec = 32
decomp_mode = config["decomp_mode"]
multivec = f"{sample}.{binsize}.E0-E{n_eigs_multivec}.{decomp_mode}.eigvecs.mv5"

CHROMSIZES = bioframe.fetch_chromsizes(assembly)

eigvals = pd.read_parquet(args.eigvals).set_index('eig')['val']
eigvecs = pd.read_parquet(args.eigvecs)

sqrt_lam = np.sqrt(np.abs(eigvals.to_numpy()))
eigvecs.loc[:, 'E0':] = (eigvecs.loc[:, 'E0':] * sqrt_lam[np.newaxis, :])

mv.to_multivec(
    multivec,
    eigvecs,
    [f'E{i}' for i in range(1, n_eigs_multivec)],
    base_res=binsize,
    chromsizes=CHROMSIZES,
)
