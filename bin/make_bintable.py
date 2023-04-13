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

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--genome', help='path to fasta file of assembly', required=True)

args = parser.parse_args()

with open(args.config, "r") as infile:
    config = yaml.full_load(infile)

assembly = config["assembly"]
binsize = config["binsize"]

CHROMSIZES = bioframe.fetch_chromsizes(assembly)
CHROMOSOMES = list(CHROMSIZES[:'chrY'].index)
CHROMOSOMES_FOR_CLUSTERING = list(CHROMSIZES[:'chr22'].index)

try:
    CENTROMERES = bioframe.fetch_centromeres(assembly)
except ValueError:
    CENTROMERES = None

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
    arms = common.make_chromarms(CHROMSIZES, mids, binsize)
arms.to_csv(
    f"{assembly}.chromarms.{binsize}.bed",
    sep='\t',
    index=False,
    header=False
)

fa_records = bioframe.load_fasta(args.genome)
df = bioframe.binnify(CHROMSIZES, binsize)
df = bioframe.frac_gc(df, fa_records)
df = common.assign_arms(df, arms)
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
    .apply(partial(common.assign_centel, arms=arms.set_index('name')))
    .reset_index(drop=True)
)
df['centel_abs'] = np.round(df['centel'] * df['armlen']).astype(int)
df.to_parquet(f"{assembly}.bins.gc.{binsize}.pq")

