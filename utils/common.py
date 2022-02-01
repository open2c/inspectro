from functools import partial
import bioframe
import numpy as np
import pandas as pd
import bbi


def split_chroms(
    df,
    points,
    cols=None,
    cols_points=None,
):
    """
    Generate a new dataframe of genomic intervals by splitting each interval 
    from the first dataframe that overlaps an interval from the second 
    dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Genomic intervals stored as a DataFrame.

    points : pandas.DataFrame or dict
        If pandas.DataFrame, a set of genomic positions specified in columns 
        'chrom', 'pos'.
        Names of cols can be overwridden by cols_points.
        If dict, mapping of chromosomes to positions.

    cols : (str, str, str) or None
        The names of columns containing the chromosome, start and end of the
        genomic intervals, provided separately for each set. The default
        values are 'chrom', 'start', 'end'.


    Returns
    -------
    df_split : pandas.DataFrame

    """
    ck1, sk1, ek1 = ['chrom', 'start', 'end']
    ck2, sk2 = ("chrom", "pos") if cols_points is None else cols_points

    name_updates = {
        ck1 + "_1": "chrom", "overlap_"+sk1: "start", "overlap_"+ek1: "end"
    }
    return_index = False
    extra_columns_1 = [i for i in list(df.columns) if i not in [ck1, sk1, ek1]]
    for i in extra_columns_1:
        name_updates[i + "_1"] = i

    if isinstance(points, dict):
        points = pd.DataFrame.from_dict(points, orient="index", columns=[sk2])
        points.reset_index(inplace=True)
        points.rename(columns={"index": "chrom"}, inplace=True)
    elif not isinstance(points, pd.DataFrame):
        raise ValueError("points must be a dict or pd.Dataframe")

    points["start"] = points[sk2]
    points["end"] = points[sk2]
    all_chroms = set(df[ck1].unique()).union(df[ck2].unique())
    all_chroms = {c:np.iinfo(np.int64).max for c in all_chroms}
    df_split = bioframe.overlap(
        df,
        bioframe.complement(
            points, 
            chromsizes=all_chroms, 
            cols=(ck2, 'start', 'end')
        ),
        how="inner",
        cols1=cols,
        cols2=(ck2, "start", "end"),
        keep_order=True,
        return_overlap=True,
        return_index=return_index,
    )[list(name_updates)]
    df_split.rename(columns=name_updates, inplace=True)
    return df_split


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
    elif group.name.endswith('q'):
        arm_start = arms.loc[this_arm, 'start']
        arm_len = arms.loc[this_arm, 'end'] - arm_start
        return (group['end'] - arm_start) / arm_len
    else:
        return group.assign(dummy=np.nan)['dummy']


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
