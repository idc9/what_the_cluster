from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from what_the_cluster.utils import _is_strictly_increasing


def get_pooled_wcss(X, class_labels):
    """
    Computes the pooled within class sum of squares for a collection
    of classes. I.e. if wcss_k is the within class sum of squares for
    the kth class (e.g. output of get_wcss) then we compute
    sum_k 1/(2 n_k) wcss_k where n_k is the number of points in the k'th class.

    Parameters
    ----------
    X (matrix) : the data set (observations on the rows)

    class_labels (list): list of class assignments (length equal to
        num observations)
    """
    num_obs = Counter(class_labels)
    wcss = get_wcss(X, class_labels)

    pooled_wcss = 0.0
    for label in num_obs.keys():
        pooled_wcss += 0.5 * (1/num_obs[label]) * wcss[label]

    return pooled_wcss


def get_wcss(X, class_labels):
    """
    Computes the within class sum of squares for a collection of classes

    sum_{i, j in C_k} ||x_j - x_i||_2^2 = 2 n_k sum_{i in C_k} ||x_i - mu_k||_2^2

    Parameters
    ----------
    X (matrix) : the data set (observations on the rows)

    class_labels (list): list of class assignments (length equal to
        num observations)

    Output
    ------
    dict keyed by cluster labels containing the within cluseter sum of squares

    """
    labels = set(class_labels)
    wcss = {}
    for label in labels:
        class_mask = class_labels == label  # which points are in this cluster
        num_points = sum(class_mask)  # how many point are in this cluster

        X_class = X[class_mask, :]
        class_mean = X_class.mean(axis=0).reshape(1, -1)

        wcss[label] = 2 * num_points * euclidean_distances(X_class,
                                                           class_mean,
                                                           squared=True).sum()

    return wcss


def estimate_n_clusters(cluster_sizes, f, se, se_adj_factor=1,
                        method='Tibs2001SEmax'):
    """
    Finds the estimates for the number of clusters using the gap statistic

    Parameters
    ---------
    cluster_sizes (list): (ordered) list of n_cluster sizes

    f (list): list of gap statistic values

    se (list): list of gap estimate standard errors

    se_adj_factor (float): adjustment factor for the estimate SEs

    method (str): which method to use to estimate the number of clusters.
    Currently one of ['firstmax', 'globalmax', 'Tibs2001SEmax']
        firstmax: finds the fist local max of f

        globalmax: finds the global max of f

        Tibs2001SEmax: uses the method detailed in (Tibshirani et al, 2001)
        i.e. the first k (smallest number of clusters) such that
        f[k] >= f[k + 1] - se[k + 1] * se_adj_factor


    Output
    ------
    est_n_clusters, possibilities

    est_n_clusters: the estimated number of clustesr
    possibilities: local maxima of the given method

    """
    assert _is_strictly_increasing(cluster_sizes)
    cluster_sizes = np.array(cluster_sizes)

    f_se_adj = se * se_adj_factor

    if method == 'firstmax':
        loc_max_inds = local_maxima(f)

        # find all local maxes
        possibilities = cluster_sizes[loc_max_inds]

        # first local max
        est_n_clusters = cluster_sizes[loc_max_inds[0]]

    elif method == 'globalmax':
        loc_max_inds = local_maxima(f)
        possibilities = cluster_sizes[loc_max_inds]
        est_n_clusters = cluster_sizes[np.argmax(f)]

    elif method == 'Tibs2001SEmax':
        f_se = f - f_se_adj

        possibilities = np.array([])
        est_n_clusters = max(cluster_sizes)
        for i in range(len(cluster_sizes) - 1):
            if f[i] >= f_se[i + 1]:
                n_clusters = cluster_sizes[i]
                possibilities = np.append(possibilities, n_clusters)

                # find the smallest number of clusters
                est_n_clusters = min(est_n_clusters, n_clusters)

    else:
        raise ValueError('%s is not a valid method' % method)

    return est_n_clusters, possibilities


def local_maxima(a):
    """
    Finds the local maxima.

    The first entry is a local max if it is greater than or equal to the second
    entry.
    The last entry is a local max if it is greater than or equal to the second
    to last entry.
    The ith entry is a local max if it is greater than or equal to both the
    i - 1 st and i + 1 st entry.

    Parameters
    ----------
    a (list): a is an array containing the sequential values of a function

    Output
    ------
    returns an array of indices corresponding to local maxima
    """
    local_max_inds = []
    for i in range(len(a)):

        is_local_max = False

        # first entry of the array
        if i == 0:
            if a[0] >= a[1]:
                is_local_max = True

        # last entry of the array
        elif i == len(a) - 1:
            if a[i] >= a[i - 1]:
                is_local_max = True

        else:
            if (a[i] >= a[i - 1]) and (a[i] >= a[i + 1]):
                is_local_max = True

        if is_local_max:
            local_max_inds.append(i)

    return local_max_inds
