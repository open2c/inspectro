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
