#!/usr/bin/env python3

from functools import partial
import bioframe
import numpy as np
import pandas as pd
import bbi


def make_chromarms(chromsizes, mids, binsize=None, suffixes=('p', 'q')):
    """
    Split chromosomes into chromosome arms
    Parameters
    ----------
    chromsizes : pandas.Series
        Series mapping chromosomes to lengths in bp.
    mids : dict-like
        Mapping of chromosomes to midpoint locations.
    binsize : int, optional
        Round midpoints to nearest bin edge for compatibility with a given
        bin grid.
    suffixes : tuple, optional
        Suffixes to name chromosome arms. Defaults to p and q.
    
    Returns
    -------
    4-column BED-like DataFrame (chrom, start, end, name).
    Arm names are chromosome names + suffix.
    Any chromosome not included in ``mids`` will be omitted.
    """
    chromosomes = [chrom for chrom in chromsizes.index if chrom in mids]

    p_arms = [
        [chrom, 0, mids[chrom], chrom + suffixes[0]]
        for chrom in chromosomes
    ]
    if binsize is not None:
        for x in p_arms:
            x[2] = int(round(x[2] / binsize)) * binsize

    q_arms = [
        [chrom, mids[chrom], chromsizes[chrom], chrom + suffixes[1]]
        for chrom in chromosomes
    ]
    if binsize is not None:
        for x in q_arms:
            x[1] = int(round(x[1] / binsize)) * binsize

    interleaved = [*sum(zip(p_arms, q_arms), ())]

    return pd.DataFrame(
        interleaved,
        columns=['chrom', 'start', 'end', 'name']
    )


def assign_arms(df, arms):
    g = {
        arm['name']: bioframe.select(
            df, (arm.chrom, arm.start, arm.end)
        ).assign(arm=arm['name']) for _, arm in arms.iterrows()
    }
    return pd.concat(g.values(), ignore_index=True)


def assign_centel(group, arms):
    this_arm = group.name
    if group.name.endswith('p'):
        arm_len = arms.loc[this_arm, 'end']
        return 1 - (group['end'] / arm_len)
    else:  # q-arm or acrocentric
        arm_start = arms.loc[this_arm, 'start']
        arm_len = arms.loc[this_arm, 'end'] - arm_start
        return (group['end'] - arm_start) / arm_len


def _fetch_binned_chrom(path, chromsizes, binsize, chrom):
    clen_rounded = int(np.ceil(chromsizes[chrom] / binsize)) * binsize
    try:
        f = bbi.open(path)
        x = f.fetch(chrom, 0, clen_rounded)
        return np.nanmean(x.reshape(-1, binsize), axis=1)
    except KeyError:
        return np.full(clen_rounded // binsize, np.nan)


def fetch_binned(path, chromsizes, chromosomes, binsize, map):
    out = map(partial(_fetch_binned_chrom, path, chromsizes, binsize), chromosomes)
    return np.concatenate(list(out))

