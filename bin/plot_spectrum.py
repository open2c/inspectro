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

args = parser.parse_args()

with open(args.config, "r") as infile:
    config = yaml.full_load(infile)

binsize = config["binsize"]
sample = config["sample"]
n_eigs = config["n_eigs"]
decomp_mode = config["decomp_mode"]
eig_pdf = f"{sample}.{binsize}.E0-E{n_eigs}.{decomp_mode}.eigvals.pdf"

# Plot the spectrum
eigval_df = pd.read_parquet(args.eigvals)
plotting.plot_spectrum(
    eigval_df,
    n_eigs_display=min(32, n_eigs),
    title= sample + "." + str(binsize),
    outpath=eig_pdf
)

plt.savefig(eig_pdf)
