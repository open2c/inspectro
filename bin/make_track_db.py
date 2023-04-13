#!/usr/bin/env python3

import matplotlib as mpl 
mpl.use("Agg")
import matplotlib.pyplot as plt 

from functools import partial
import os.path as op
import pathlib

from tqdm import tqdm
from loky import get_reusable_executor
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
parser.add_argument('--bins', help='parquet bins file', required=True)
parser.add_argument('--meta', help='metadata file', required=True)

args = parser.parse_args()

with open(args.config, "r") as infile:
    config = yaml.full_load(infile)
    
assembly = config["assembly"]
binsize = config["binsize"]
track_db = f"tracks.{assembly}.{binsize}.h5"

CHROMSIZES = bioframe.fetch_chromsizes(assembly)
CHROMOSOMES = list(CHROMSIZES[:'chrY'].index)
CHROMOSOMES_FOR_CLUSTERING = list(CHROMSIZES[:'chr22'].index)

try:
    CENTROMERES = bioframe.fetch_centromeres(assembly)
except ValueError:
    CENTROMERES = None

h5opts = dict(compression='gzip', compression_opts=6)

bins = pd.read_parquet(args.bins)

if not op.exists(track_db):
    with h5py.File(track_db, 'w') as f:
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

meta = pd.read_table(args.meta)
paths = meta.set_index('ID')['Path']
with h5py.File(track_db, 'a') as f:
    for ix, row in meta.iterrows():
        if row['ID'] in f:
            continue

        if row['FileFormat'].lower() == 'bigwig':
            with get_reusable_executor(26) as pool:
                acc = row['ID']
                x = common.fetch_binned(
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


