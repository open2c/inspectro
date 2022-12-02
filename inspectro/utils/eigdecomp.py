import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats

from cooltools.lib import numutils
from cooltools.api.eigdecomp import _filter_heatmap, _fake_cis

import bioframe
import cooler


def _orient_eigs(eigvecs, phasing_track, corr_metric=None):
    """
    Orient each eigenvector deterministically according to the orientation
    that correlates better with the phasing track.

    Parameters
    ----------
    eigvecs : 2D array (n, k)
        `k` eigenvectors (as columns).
    phasing_track : 1D array (n,)
        Reference track for determining orientation.
    corr_metric: spearmanr, pearsonr, var_explained, MAD_explained
        Correlation metric to use for selecting orientations.

    Returns
    -------
    2D array (n, k)
        Reoriented `k` eigenvectors.

    Notes
    -----
    This function does NOT change the order of the eigenvectors.
    """
    for i in range(eigvecs.shape[1]):
        
        mask = np.isfinite(eigvecs[:, i]) & np.isfinite(phasing_track)

        if corr_metric is None or corr_metric == "spearmanr":
            corr = scipy.stats.spearmanr(phasing_track[mask], eigvecs[mask, i])[0]
        elif corr_metric == "pearsonr":
            corr = scipy.stats.pearsonr(phasing_track[mask], eigvecs[mask, i])[0]
        elif corr_metric == "var_explained":
            corr = scipy.stats.pearsonr(phasing_track[mask], eigvecs[mask, i])[0]
            # multiply by the sign to keep the phasing information
            corr = np.sign(corr) * corr * corr * np.var(eigvecs[mask, i])
        elif corr_metric == "MAD_explained":
            corr = (
                numutils.COMED(phasing_track[mask], eigvecs[mask, i]) *
                numutils.MAD(eigvecs[mask, i])
            )
        else:
            raise ValueError("Unknown correlation metric: {}".format(corr_metric))

        eigvecs[:, i] = np.sign(corr) * eigvecs[:, i]

    return eigvecs


def _normalized_affinity_matrix_from_trans(A, partition, perc_top, perc_bottom):
    """
    Produce an affinity matrix based on trans data by filling in cis regions
    with randomly sampled trans pixels from the same row or column.

    The resulting matrix is rebalanced and uniformly scaled such that all rows
    and columns sum to 1 (a.k.a. a stochastic matrix),

    Parameters
    ----------
    A : 2D array (n, n)
        Whole genome contact matrix.
    partition : 1D array (n_chroms+1,)
        An offset array providing the starting bin of each chromosome and
        whose last element is the last bin of the last chromosome.
    perc_top : float
        Clip trans blowout pixels above this cutoff.
    perc_bottom : 
        Mask bins with trans coverage below this cutoff.
    
    Returns
    -------
    2D array (n, n)
        Normalized affinity matrix
    """
    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not symmetric")

    n_bins = A.shape[0]
    if not (
        partition[0] == 0 and partition[-1] == n_bins and np.all(np.diff(partition) > 0)
    ):
        raise ValueError(
            "Not a valid partition. Must be a monotonic sequence "
            "from 0 to {}.".format(n_bins)
        )

    # Zero out cis data and create mask for trans
    extents = zip(partition[:-1], partition[1:])
    part_ids = []
    for n, (i0, i1) in enumerate(extents):
        A[i0:i1, i0:i1] = 0
        part_ids.extend([n] * (i1 - i0))
    part_ids = np.array(part_ids)
    is_trans = part_ids[:, None] != part_ids[None, :]

    # Zero out bins nulled out using NaNs
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    if np.any(~np.isfinite(A)):
        raise ValueError("Matrix A contains point-wise NaNs or infinite values, not expected for cooler-loaded maps")

    # Filter the heatmap
    is_good_bin = ~is_bad_bin
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    A = _filter_heatmap(A, is_trans & is_valid, perc_top, perc_bottom)
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Inject decoy cis data, balance and rescale margins to 1
    A = _fake_cis(A, ~is_trans)
    numutils.set_diag(A, 0, 0)
    A = numutils.iterative_correction_symmetric(A)[0]
    marg = np.r_[np.sum(A, axis=0), np.sum(A, axis=1)]
    marg = np.mean(marg[marg > 0])
    A /= marg

    A = _fake_cis(A, ~is_trans)
    numutils.set_diag(A, 0, 0)
    A = numutils.iterative_correction_symmetric(A)[0]
    marg = np.r_[np.sum(A, axis=0), np.sum(A, axis=1)]
    marg = np.mean(marg[marg > 0])
    A /= marg

    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    return A


def eig_trans(
    clr,
    bins,
    n_eigs=3,
    partition=None,
    balance="weight",
    perc_bottom=1,
    perc_top=99.95,
    phasing_track_col="GC",
    corr_metric=None,
    which='LM',
):
    """
    Spectral decomposition of trans Hi-C data derived from a normalized 
    affinity representation.

    Each eigenvector is deterministically oriented with respect to a provided 
    "phasing track" (e.g. GC content).

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object.
    bins : DataFrame
        Cooler-compatible bin table with phasing track column.
        If a column named "is_bad" is present, bins with nonzero values will
        filtered out along with those having NaN balancing weights.
    n_eigs : int
        Number of eigenvectors to calculate, after E0.
    partition : 1D array (n_chroms + 1,), optional
        An offset array providing the starting bin of each chromosome and
        whose last element is the last bin of the last chromosome.
    balance : str or bool
        Name of weight column to use for balancing. If True, use the default
        name "weight". If False, do not balance the raw contact matrix.
    perc_top : float
        Clip trans blowout pixels above this cutoff.
    perc_bottom : float
        Mask bins with trans coverage below this cutoff.
    phasing_track_col : 
        Column of bin table to use for deterministically orienting the 
        eigenvectors.
    corr_metric : str
        Correlation metric to use for selecting orientations.
    which : str
        Code for the eigenvalue order in which components are calculated.
        (LM = largest/descending magnitude/modulus; LA = largest/descending 
        algebraic value).

    Returns
    -------
    eigvals : DataFrame (n_eigs + 1, 2)
        Table of eigenvalues.
    eigvecs : DataFrame (n, n_eigs + 1)
        Table of eigenvectors (as columns).

    Notes
    -----
    This is very similar to the trans eigendecomposition method from 
    Imakaev et al. 2012 and the implementation in cooltools but differs in 
    how the matrix is normalized before being decomposed. The main impact is 
    that the eigen*value* spectra end up being standardized and thus easier to 
    assess and compare between datasets. Moreover, because the matrix is not
    mean-centered before decomposition, an additional trivial eigenvector will 
    be produced having eigenvalue 1. Hence, we return n_eigs + 1 vectors.

    For more details, see the Supplemental Note of Spracklin, Abdennur et al.,
    2021: https://www.biorxiv.org/content/10.1101/2021.08.05.455340v1.supplementary-material

    """
    if partition is None:
        partition = np.r_[
            [clr.offset(chrom) for chrom in clr.chromnames], len(clr.bins())
        ]
    lo = partition[0]
    hi = partition[-1]

    A = clr.matrix(balance=balance)[lo:hi, lo:hi]
    bins = bins[lo:hi]

    # Apply blacklist if available.
    if 'is_bad' in bins.columns:
        mask = bins['is_bad'].values.astype(bool)
        A[mask, :] = np.nan
        A[:, mask] = np.nan

    # Extract phasing track.
    phasing_track = None
    if phasing_track_col:
        if phasing_track_col not in bins:
            raise ValueError(
                'No column "{}" in the bin table'.format(phasing_track_col)
            )
        phasing_track = bins[phasing_track_col].values[lo:hi]

    # Compute the affinity matrix.
    A = _normalized_affinity_matrix_from_trans(
        A, partition, perc_top, perc_bottom
    )

    # Compute eigs on the doubly stochastic affinity matrix
    # We actually extract n + 1 eigenvectors.
    # The first eigenvector, E0, will be uniform with eigenvalue 1.
    mask = np.sum(np.abs(A), axis=0) != 0
    A_collapsed = A[mask, :][:, mask].astype(np.float, copy=True)
    eigvals, eigvecs_collapsed = scipy.sparse.linalg.eigsh(
        A_collapsed,
        n_eigs + 1,
        which=which
    )
    eigvecs = np.full((len(mask), n_eigs + 1), np.nan)
    eigvecs[mask, :] = eigvecs_collapsed

    # Ensure order by descending |eigval|
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Reorient the vectors deterministically.
    if phasing_track is not None:
        eigvecs = _orient_eigs(eigvecs, phasing_track, corr_metric)

    # Prepare outputs.
    eigval_table = pd.DataFrame({
        'eig': ["E{}".format(i) for i in range(n_eigs + 1)],
        'val': eigvals,
    })
    eigvec_table = bins.copy()
    for i in range(n_eigs + 1):
        eigvec_table["E{}".format(i)] = eigvecs[:, i].copy()

    return eigval_table, eigvec_table


def _normalized_affinity_matrix_from_cis(A, perc_top, perc_bottom, ignore_diags=2):
    """
    Produce an affinity matrix based on cis data.

    The resulting matrix is rebalanced and uniformly scaled such that all rows
    and columns sum to 1 (a.k.a. a stochastic matrix).

    Parameters
    ----------
    A : 2D array (n, n)
        Whole genome contact matrix.
    perc_top : float
        Clip trans blowout pixels above this cutoff.
    perc_bottom :
        Mask bins with trans coverage below this cutoff.
    ignore_diags :
        Number of diagonals to ignore, 2 by default

    Returns
    -------
    2D array (n, n)
        Normalized affinity matrix
    """

    # Input checks
    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not symmetric")

    n_bins = A.shape[0]
    if ignore_diags >= n_bins:
        raise ValueError("Number of ignored diagonals should be less than the number of bins")

    # Zero out bins nulled out using NaNs
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    if np.any(~np.isfinite(A)):
        raise ValueError("Matrix A contains point-wise NaNs or infinite values, not expected for cooler-loaded maps")

    # Additional checks for ignore_diags after
    is_good_bin = ~is_bad_bin

    if A.shape[0] <= ignore_diags + 3 or is_good_bin.sum() <= ignore_diags + 3:
        return np.nan * np.ones(A.shape)

    if ignore_diags:
        for d in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(A, 1.0, d)

    # Zero out bins nulled out using NaNs
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Filter the heatmap
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    A = _filter_heatmap(A, is_valid, perc_top, perc_bottom)
    is_bad_bin = np.nansum(A, axis=0) == 0
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Convert to observed over expected:
    OE, _, _, _ = numutils.observed_over_expected(A, is_good_bin)

    # Inject zero diagonal, balance and rescale margins to 1
    numutils.set_diag(A, 0, 0)
    OE = numutils.iterative_correction_symmetric(OE)[0]
    marg = np.r_[np.sum(OE, axis=0), np.sum(OE, axis=1)]
    marg = np.mean(marg[marg > 0])
    OE /= marg

    # empty invalid rows, so that get_eig can find them
    OE[is_bad_bin, :] = 0
    OE[:, is_bad_bin] = 0

    return OE

def eig_cis(
    clr,
    bins,
    n_eigs=3,
    view_df=None,
    ignore_diags=None,
    balance="weight",
    perc_bottom=1,
    perc_top=99.95,
    phasing_track_col="GC",
    corr_metric=None,
    which='LM',
):
    """
    Spectral decomposition of cis Hi-C data derived from a normalized
    affinity representation.

    Each eigenvector is deterministically oriented with respect to a provided
    "phasing track" (e.g. GC content).

    clr : cooler.Cooler
        Cooler object.
    bins : DataFrame
        Cooler-compatible bin table with phasing track column.
        If a column named "is_bad" is present, bins with nonzero values will
        be filtered out along with those having NaN balancing weights.
    n_eigs : int
        Number of eigenvectors to calculate, after E0.
    view_df: bioframe.viewframe
        Viewframe for extraction of the cis maatrices
    ignore_diags: int
        Number of diagonals to ignore, by default loaded from cooler
    balance : str or bool
        Name of weight column to use for balancing. If True, use the default
        name "weight". If False, do not balance the raw contact matrix.
    perc_top : float
        Clip trans blowout pixels above this cutoff.
    perc_bottom : float
        Mask bins with trans coverage below this cutoff.
    phasing_track_col :
        Column of bin table to use for deterministically orienting the
        eigenvectors.
    corr_metric : str
        Correlation metric to use for selecting orientations.
    which : str
        Code for the eigenvalue order in which components are calculated.
        (LM = largest/descending magnitude/modulus; LA = largest/descending
        algebraic value).

    Returns
    -------
    eigvals : DataFrame (n_eigs + 1, 2)
        Table of eigenvalues.
    eigvecs : DataFrame (n, n_eigs + 1)
        Table of eigenvectors (as columns).

    Notes
    -----
    This is adaptaion of the decomposition technique from eig_trans to cis matrices.
    The key difference is that cis-data is not masked out with random noise,
    the main diagonal is filled in with zeros.

    """

    chomosomes_bins = np.unique(bins.chrom)

    # Verify or create view_df:
    # get chromosomes from cooler, if view_df not specified:
    if view_df is None:
        view_df = bioframe.make_viewframe(
            [(chrom, 0, clr.chromsizes[chrom]) for chrom in clr.chromnames and chrom in chomosomes_bins]
        )
    else:
        # appropriate viewframe checks:
        if not bioframe.is_viewframe(view_df):
            raise ValueError("view_df is not a valid viewframe.")
        if not bioframe.is_contained(view_df, bioframe.make_viewframe(clr.chromsizes)):
            raise ValueError("view_df is out of the bounds of chromosomes in cooler.")

    # Make sure phasing_track_col is in bins, if phasing is requested
    if phasing_track_col and (phasing_track_col not in bins):
        raise ValueError(f'No column "{phasing_track_col}" in the bin table')

    # Ignore diags as in cooler inless specified
    ignore_diags = (
        clr._load_attrs("bins/weight").get("ignore_diags", 2)
        if ignore_diags is None
        else ignore_diags
    )

    # Prepare output table with eigenvectors
    eigvec_table = bins.copy()
    eigvec_columns = [f"E{i}" for i in range(n_eigs+1)]
    for ev_col in eigvec_columns:
        eigvec_table[ev_col] = np.nan

    # Prepare output table with eigenvalues
    eigval_table = pd.DataFrame({
        'eig': eigvec_columns
    })

    def _each(region):
        """
        perform eigen decomposition for a given region
        assuming safety checks are done outside of this
        function.
        Parameters
        ----------
        region: tuple-like
            tuple of the form (chroms,start,end,*)
        Returns
        -------
        _region, eigvals, eigvecs -> ndarrays
            array of eigenvalues and an array eigenvectors
        """

        _region = region[:3]  # take only (chrom, start, end)
        A = clr.matrix(balance=balance).fetch(_region)

        # Apply blacklist if available.
        if 'is_bad' in bins.columns:
            mask = bioframe.select(bins, _region)['is_bad'].values.astype(bool)
            A[mask, :] = np.nan
            A[:, mask] = np.nan

        A = _normalized_affinity_matrix_from_cis(A, perc_top, perc_bottom, ignore_diags)

        # Extract phasing track relevant for the _region
        phasing_track = (
            bioframe.select(bins, _region)[phasing_track_col].values
            if phasing_track_col
            else None
        )

        # Compute eigs on the doubly stochastic affinity matrix
        # We actually extract n + 1 eigenvectors.
        # The first eigenvector, E0, will be uniform with eigenvalue 1.
        mask = np.sum(np.abs(A), axis=0) != 0

        if (A.shape[0] <= ignore_diags + n_eigs + 1) or (mask.sum() <= ignore_diags + n_eigs + 1):
            return _region, np.nan * np.ones(n_eigs+1), np.nan * np.ones((len(A), n_eigs+1))

        A_collapsed = A[mask, :][:, mask].astype(np.float, copy=True)
        eigvals, eigvecs_collapsed = scipy.sparse.linalg.eigsh(
            A_collapsed,
            n_eigs + 1,
            which=which
        )
        eigvecs = np.full((len(mask), n_eigs + 1), np.nan)
        eigvecs[mask, :] = eigvecs_collapsed

        # Ensure order by descending |eigval|
        order = np.argsort(-np.abs(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if phasing_track is not None:
            eigvecs = _orient_eigs(eigvecs, phasing_track, corr_metric)

        return region[:4], eigvals, eigvecs

    # Eigendecompose matrix per region
    # Output assumes that the order of results matches regions
    results = map(_each, view_df.values)

    # Go through eigendecomposition results and fill in
    # Output table eigvec_table and eigvals_table
    for _region, _eigvals, _eigvecs in results:

        if len(_region)==3:
            region_name = f'{_region[0]}:{_region[1]}-{_region[2]}'
        else:
            region_name = _region[3]

        idx = bioframe.select(eigvec_table, _region).index.values
        eigvec_table.loc[idx, 'name'] = region_name
        eigvec_table.loc[idx, eigvec_columns] = _eigvecs

        eigval_table.loc[:, region_name] = _eigvals

    return eigval_table, eigvec_table



def projection(A, eigs, n_eigs=None):
    """
    Project matrix A to the eigenvectors eigs.

    A : Input matrix
    eigs : eigenvector DataFrame
    n_eigs : limiting number of eigenvectors

    Returns
    -------
    proj_table -> DataFrame (n, n_eigs + 1)
        Table of projected values (as columns).

    """

    # Input check:
    n_bins = A.shape[0]
    if n_bins != len(eigs):
        raise ValueError(f"Matrix and eigs shape mismatch: {n_bins} {len(eigs)}")

    # Filter out missing data:
    E = eigs.loc[:, 'E0':f'E{n_eigs}'].to_numpy().astype(np.float64)
    mask = (
            (np.sum(np.abs(A), axis=0) != 0) &
            (np.sum(np.isnan(E), axis=1) == 0)
    )
    A_collapsed = A[mask, :][:, mask].astype(float, copy=True)
    E_collapsed = E[mask, :]

    # Project:
    proj = np.full((n_bins, n_eigs + 1), np.nan)
    result = []
    for i in range(n_eigs + 1):
        result.append(np.dot(A_collapsed, E_collapsed[:, i][:, np.newaxis]))
    proj[mask, :] = np.hstack(result)

    # Create projection table:
    proj_table = eigs.loc[:, ['chrom', 'start', 'end']].copy()
    for i in range(n_eigs + 1):
        proj_table["E{}".format(i)] = proj[:, i].copy()
        proj_table["E{}".format(i)] /= np.linalg.norm(proj_table["E{}".format(i)].dropna())

    return proj_table