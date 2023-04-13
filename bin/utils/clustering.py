#!/usr/bin/env python3

import warnings
import bioframe
import numpy as np
import pandas as pd


def _extract_eigs(
    eigvals, 
    eigvecs, 
    n_clusters, 
    n_components, 
    weight_by_eigval, 
    keep_first, 
    positive_eigs=False, 
    filter_nans=True
):
    eigvecs = eigvecs.copy()

    # Decide whether to use unit normed vectors or to weight them by sqrt(|lambda_i|)
    if weight_by_eigval:
        eigvecs.loc[:, 'E0':] *= np.sqrt(np.abs(eigvals.T.values))

    # Do the k-clustering on top k eigenvectors, unless overriden to use more or fewer
    if n_components is None:
        n_components = n_clusters

    if not positive_eigs:
        # Decide whether to use E0 or not
        if keep_first:
            elo, ehi = 'E0', f'E{n_components - 1}'
        else:
            elo, ehi = 'E1', f'E{n_components}'
        X = eigvecs.loc[:, elo:ehi].values
    else:
        if not keep_first:
            eigvals = eigvals.drop('E0')
        which = eigvals.loc[eigvals['val'] > 0].index[:n_components]
        X = eigvecs.loc[:, which].values

    if not filter_nans:
        return X

    mask = np.all(~np.isnan(X), axis=1)
    x = X[mask, :]

    return x, mask


def relabel_clusters(labels, n_clusters, sorting_tracks, sort_key):
    """
    Re-order the bins and re-label the cluster IDs based on a set of
    sorting tracks.
    1. User-defined sorting key.
    2. Absolute distance from centromere.
    3. Length of corresponding chromosome arm.
    """
    # Assign the cluster IDs and extra data to temporary dataframe
    df = sorting_tracks[['chrom', 'start', 'end', sort_key, 'centel', 'armlen']].copy()
    df['centel_abs'] = df['centel'] * df['armlen']
    df['cluster'] = labels

    # Relabel the clusters using median of sorting column
    df.loc[df['cluster'] == n_clusters, sort_key] = np.inf
    clusters_ordered = (
        df
        .groupby('cluster')
        [sort_key]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    cluster_dtype = pd.CategoricalDtype(clusters_ordered, ordered=True)
    df['cluster_relabeled'] = df['cluster'].astype(cluster_dtype).cat.codes

    # Reorder the bins for plotting
    bin_ranks = (
        df
        .sort_values(
            ['cluster_relabeled', 'centel_abs'],
            ascending=[True, True]
        )
        .index
        .values
    )
    return df['cluster_relabeled'].values, bin_ranks


def kmeans_sm(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Shi and Malik (2000)
    * Use the eigenvectors of L_rw (this doesn't matter for us).
    * NO row normalization before clustering.
    Notes
    -----
    This is what sklearn spectral_clustering does on the eigenvectors of the 
    normalized laplacian (when using the k-means method).
    Sklearn's implementation does not unit norm the input vectors as some do.
    """
    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=100,
        max_iter=10000,
        tol=0.00001,
        # precompute_distances='auto',
        # verbose=0,
        random_state=42,
        copy_x=True,
        # n_jobs=32,
        # algorithm='auto',
    )

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )
    labels = np.full(len(mask), n_clusters)
    labels[mask] = model.fit_predict(x)

    return labels


def gaussian_mixture_sm(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Extends the Shi and Malik method to a GMM with no covariance constraints.
    * Use the eigenvectors of L_rw (this doesn't matter for us).
    * NO row normalization before clustering.
    """

    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_gaussian_mixture.py
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        tol=0.001,
        reg_covar=1e-06,
        n_init=100,
        max_iter=1000,
        init_params='kmeans',
        random_state=42,
        verbose=0
    )

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )
    labels = np.full(len(mask), n_clusters)
    labels[mask] = model.fit_predict(x)

    if not model.converged_:
        warnings.warn(f"GMM did not converge for k={n_clusters}")

    return labels


def kmeans_njw(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Ng, Jordan and Weiss (2002)
    * Use the eigenvectors of L_sym (this doesn't matter for us).
    * Normalize the rows to norm 1.
    """
    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=100,
        max_iter=10000,
        tol=0.00001,
        verbose=0,
        random_state=42,
        n_jobs=32,
        algorithm='auto',
    )

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )

    # Normalize rows to norm 1
    y = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

    labels = np.full(len(mask), n_clusters)
    labels[mask] = model.fit_predict(y)

    return labels


def gaussian_mixture_njw(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Extends the Shi and Malik method to a GMM with no covariance constraints.
    * Use the eigenvectors of L_rw (this doesn't matter for us).
    * Normalize the rows to norm 1.
    """

    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_gaussian_mixture.py
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        tol=0.001,
        reg_covar=1e-06,
        n_init=100,
        max_iter=1000,
        init_params='kmeans',
        random_state=42,
        verbose=0
    )

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )

    # Normalize rows to norm 1
    y = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

    labels = np.full(len(mask), n_clusters)
    labels[mask] = model.fit_predict(y)

    if not model.converged_:
        warnings.warn(f"GMM did not converge for k={n_clusters}")

    return labels


def discretize_ys(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Yu and Shi (2003)
    * Use the unit-norm eigenvectors of L_rw (this doesn't matter for us).
    * Normalize the rows to norm 1.
    * Find the closest discrete partition matrix (one-hot rows) to X (minimize Ncut loss)
    Notes
    -----
    This is what sklearn's spectral_clustering uses with assign_labels='discretize'.
    `weight_by_eigval` won't work because `discretize` internally renormalizes
    the eigenvectors to norm 1.
    References
    ----------
    https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf
    """

    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_spectral.py#L23
    from sklearn.cluster._spectral import discretize

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )
    labels = np.full(len(mask), n_clusters)

    # Sklearn does the row normalization and initial orientation of eigs
    labels[mask] = discretize(x, random_state=42)

    return labels


def spherical_kmeans(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Project points onto the unit hypersphere and cluster.
    * Use the unit-norm eigenvectors of L_rw or L_sym (doesn't matter for us).
    * Normalize the *rows* to norm 1 to project points onto the surface of the unit hypersphere.
      (technically, spherical k-means should do the projection anyway)
    * Do spherical k-means
    Notes
    -----
    https://stackoverflow.com/a/38900937
    """

    # https://github.com/jasonlaska/spherecluster
    from spherecluster import SphericalKMeans

    model = SphericalKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=100,
        max_iter=10000,
        tol=0.00001,
        verbose=0,
        random_state=42,
        n_jobs=32,
    )

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )

    # Normalize rows to norm 1
    y = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

    labels = np.full(len(mask), n_clusters)
    labels[mask] = model.fit_predict(y)

    return labels


def vonmises_mixture(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Extend the spherical kmeans method to a von Mises-Fisher (gaussian on a sphere) MM.
    * Use the unit-norm eigenvectors of L_rw or L_sym (doesn't matter for us).
    * Normalize the *rows* to norm 1 to project points onto the surface of the unit hypersphere.
      (technically, spherical k-means should do the projection anyway)
    * Do the mixture fitting
    Notes
    -----
    Unlike the regular gaussian mixture, the distributions in this model are isotropic.
    """
    from spherecluster import VonMisesFisherMixture

    model = VonMisesFisherMixture(
        n_clusters=n_clusters,
        posterior_type="soft",
        n_init=100,
        max_iter=1000,
        init="random-class",
        random_state=42,
        tol=1e-6,
        normalize=True,
        verbose=False,
        n_jobs=32,
    )

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )

    # Normalize rows to norm 1
    y = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

    labels = np.full(len(mask), n_clusters)
    labels[mask] = model.fit_predict(y)

    # posterior = model.posterior_   # [n_clusters, n_examples]
    return labels


def gaussian_hmm(eigvals, eigvecs, n_clusters, n_components=None, weight_by_eigval=False, keep_first=True, positive_eigs=False):
    """
    Model the rows (points) as an HMM on k different latent k-dimensional 
    Gaussian states.
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.cluster import KMeans

    x, mask = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, positive_eigs
    )
    NULL = -100
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=100,
        max_iter=10000,
        tol=0.00001,
        precompute_distances='auto',
        verbose=0,
        random_state=42,
        copy_x=True,
        n_jobs=32,
        algorithm='auto',
    )
    kmeans.fit(x)
    means_init = kmeans.cluster_centers_.tolist()
    means_init.append([NULL] * n_components)
    means_init = np.array(means_init)

    X = _extract_eigs(
        eigvals, eigvecs, n_clusters, n_components, weight_by_eigval, keep_first, filter_nans=False, positive_eigs=positive_eigs
    )
    X[np.isnan(X)] = NULL
    hmm = GaussianHMM(
        n_components=n_clusters + 1,  # add 1 explicit NULL state
        covariance_type='full',
        min_covar=0.0001,
        n_iter=100,
        random_state=42,
        init_params='stc',
    )
    hmm.means_ = means_init
    hmm.fit(X)

    Y = hmm.predict(X)
    key = -hmm.means_.min(axis=1)
    relabel = dict(zip(
        range(n_clusters + 1),
        np.argsort(key)
    ))
    labels = np.array([relabel[y] for y in Y])

    return labels

