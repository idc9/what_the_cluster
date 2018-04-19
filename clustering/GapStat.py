from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

from sklearn.externals import joblib

from .gapstat_utils import get_pooled_wcss, estimate_n_clusters
from .reference_dists import sample_svd_null, sample_uniform_null
from .utils import _is_strictly_increasing, _count_none, svd_wrapper
from .clusterers import get_clusterer

# TODO: implement seeds
# TODO: give clusterer the option to return additional data
class GapStat(object):

    def __init__(self,
                 clusterer='kmeans',
                 clusterer_kwargs={},
                 cluster_sizes=list(range(1, 11)),
                 ref_dist='uniform',
                 B=10,
                 gap_est_method='Tibs2001SEmax'):
        """

        For details see Estimating the Number of Clusters in a Data Set via
        the Gap Statistic by R. Tibshirani, G. Walther and T. Hastie, 2001.

        Parameters
        ----------
        clusterer (str, function): a function which computes clusters.
            If clusterer is a string, the will used one of the pre-implemented
            clustering algorithms from clusterers.py. Available options include
            ['kmeans']

            If clusterer is a function then it should accpet two argumets:
            (X, n_clusters) where X is the data set to cluster and n_clusters
            is the number of desired clusters to estimate. This function should
            return a list of estimated clusters for each observation.

        clusterer_kwargs (None, dict): dict of key word arguments for the
            clusterer function. See the documentation for the orignal functions
            for available arguments (linked to in clusterers.py)

            Warning: these are only applied for the
            pre-implemented clusterers i.e. if clusterer is a string.


        cluster_sizes (list): list of n_clusters to evaluate. Must be
            strictly increasing.

        ref_dist (str): which null reference distribution to use. Either
            ['uniform', 'svd']. 'uniform' will draw uniform smaples
            from a box which has the same range of the data. 'PCA' will
            use the prinicpal components to better adapt the shape of
            the reference distribution to the observed data set.
            See (Tibshirani et al, 2001) for details.

        B (int): number of samples of null reference set to draw to estimated
            the E log(W)

        gap_est_method (str): how to select the local max using the gap
            statistic. Currently one of ['firstmax', 'globalmax',
            'Tibs2001SEmax']. See estimate_n_clusters() for details.
        """
        assert ref_dist in ['uniform', 'svd']
        assert _is_strictly_increasing(cluster_sizes)

        self.ref_dist = ref_dist
        self.B = B
        self.cluster_sizes = cluster_sizes
        self.gap_est_method = gap_est_method

        if callable(clusterer):
            # there might be an issue with python 3.x for x <2
            # see https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
            self.clusterer_name = 'custom'
            self.clusterer = clusterer

            if clusterer_kwargs is not None:
                # TODO: make this a proper Warning
                print("WARNING: clusterer_kwargs is only use for pre-implemented clusterers")
        else:
            self.clusterer_name = clusterer

            if clusterer == 'custom':
                # this means we are loading a saved version of this object
                # and we didn't save the clusterer funciton which should be
                # saved separately
                self.clusterer = None
            else:
                self.clusterer = get_clusterer(clusterer, clusterer_kwargs)

        # only store this in case we save this object to disk
        self.clusterer_kwargs = clusterer_kwargs

        # these attributes will be set later
        self.X = None  # observed data
        self.U = None  # U, D, V are SVD of X
        self.D = None
        self.V = None

        self.obs_cluster_labels = None
        self.obs_wcss = None
        self.null_wcss_samples = None
        self.est_n_clusters = None
        self.possible_n_clusters = None

    def estimate_n_clusters(self, X, cluster_labels=None,
                            U=None, D=None, V=None):
        """
        Estimates the number of clusters using the gap statistic.

        Parameters
        ----------
        X (matrix): the observed data

        cluster_labels (None or matrix, observations x len(cluster_sizes)): matrix
            containing the observed cluster labels on the columns for each
            value of n_clusters.

            If None then will uses clusterer to estimate the number of clusters
            using the provided clusterer


        U, D, V: the precomputed SVD of X see set_svd_decomposition() for
            details. These are only used if ref_dist = 'svd'. If they are not
            provided then will compute them.
        """
        if cluster_labels is None:
            self.compute_obs_clusters(X)
        else:
            self.set_obs_clusters(X, cluster_labels)

        if self.ref_dist == 'svd':
            if _count_none(U, D, V) == 0:
                self.set_svd_decomposition(U, D, V)

            elif _count_none(U, D, V) == 3:
                self.compute_svd_decomposition()

            else:
                raise ValueError('U, D, V must all be provided or be set to None')

        self.compute_obs_wcss()
        self.sample_ref_null_wcss()
        self.compute_n_cluster_estimate(method=self.gap_est_method)
        # return self.est_n_clusters # I think we don't want to return anything

    def set_obs_clusters(self, X, cluster_labels):
        """

        Parameters
        ----------
        X (matrix): the observed data

        cluster_labels (matrix, observations x len(cluster_sizes)): matrix
            containing the observed cluster labels on the columns for each
            value of n_clusters
        """
        assert cluster_labels.shape == (X.shape[0], len(self.cluster_sizes))

        self.X = X
        self.obs_cluster_labels = cluster_labels

    def compute_obs_clusters(self, X):

        obs_cluster_labels = np.zeros((X.shape[0], len(self.cluster_sizes)))

        for i, n_clusters in enumerate(self.cluster_sizes):
            obs_cluster_labels[:, i] = self.clusterer(X, n_clusters)

        self.set_obs_clusters(X, obs_cluster_labels)

    def set_svd_decomposition(self, U, D, V):
        """
        Stores the SVD decomposition of X which is only used for the SVD
        null reference distribution. U, D, V are the scores, singluar
        values and loadings respectively. The user may have already commputed
        the SVD of the observed data and can use this method to set it for
        the GapStat object.

        If X is a (n x d) matrix and r = min(n, d)
            U must b a (n x r) matrix
            D must be a list of length r
            V must be a (d x r) matrix
        D, and the columns of U, D must be sorted according to decreasing
        singular values. See utils.svd_wrapper


        Parameters
        ----------
        U, D, V are matrices described as above
        """
        assert U.shape == (self.X.shape[0], min(self.X.shape))
        assert len(D) == min(self.X.shape)
        assert V.shape == (self.X.shape[1], min(self.X.shape))

        self.U = U
        self.D = D
        self.V = V

    def compute_svd_decomposition(self):
        U, D, V = svd_wrapper(self.X)
        self.set_svd_decomposition(U, D, V)

    def compute_obs_wcss(self):
        """
        Computes the within class sum of squres for the observed clusters.
        """
        n_cluster_sizes = len(self.cluster_sizes)
        self.obs_wcss = np.zeros(n_cluster_sizes)

        for j in range(n_cluster_sizes):
            # make sure the number of unique cluster labels is equal to
            # the preported number of clusters
            # TODO: we might not want this restrictin
            assert len(set(self.obs_cluster_labels[:, j])) \
                == self.cluster_sizes[j]

            self.obs_wcss[j] = get_pooled_wcss(self.X,
                                               self.obs_cluster_labels[:, j])

    def _sample_null_reference(self):

        if self.ref_dist == 'uniform':
            return sample_uniform_null(self.X)
        elif self.ref_dist == 'svd':
            return sample_svd_null(self.X, self.U, self.D, self.V)

    def sample_ref_null_wcss(self):

        self.null_wcss_samples = np.zeros((len(self.cluster_sizes), self.B))

        for b in range(self.B):
            # sample null reference distribution
            X_null = self._sample_null_reference(b)

            # cluster X_null for the specified n_clusters
            for i, n_clusters in enumerate(self.cluster_sizes):
                # cluster. null sample
                null_cluster_labels = self.clusterer(X_null, n_clusters)

                self.null_wcss_samples[i, b] = get_pooled_wcss(X_null,
                                                               null_cluster_labels)

    @property
    def E_log_null_wcss_est(self):
        """
        Estimate of the expected log(WCSS) of the null reference distribution
        """
        assert self.null_wcss_samples is not None
        return np.log(self.null_wcss_samples).mean(axis=1)

    @property
    def E_log_null_wcss_est_sd(self):
        """
        Standard deviation of the estimated expected log(WCSS) from the null
        distribuiton
        """
        assert self.null_wcss_samples is not None
        return np.std(np.log(self.null_wcss_samples), axis=1)

    @property
    def log_obs_wcss(self):
        """
        log(WCSS) of the observed cluseters
        """
        assert self.obs_wcss is not None
        return np.log(self.obs_wcss)

    @property
    def gap(self):
        """
        Returns the gap statistic i.e. E*(log(WCSS_null)) - log(WCSS_obs)
        where E* means the estimated expected value
        """
        assert self.obs_wcss is not None

        return self.E_log_null_wcss_est - self.log_obs_wcss

    @property
    def adj_factor(self):
        return sqrt(1.0 + (1.0/self.B))

    def compute_n_cluster_estimate(self, method=None):
        """
        Parameters
        ----------
        method (str): which method to use to estimate the number of clusters.
        Currently one of ['firstmax', 'globalmax', 'Tibs2001SEmax']
            firstmax: finds the fist local max of f

            globalmax: finds the global max of f

            Tibs2001SEmax: uses the method detailed in (Tibshirani et al, 2001)
            i.e. the first k (smallest number of clusters) such that
            f[k] >= f[k + 1] - se[k + 1] * se_adj_factor

        return_possibilities (bool): whether or not to also return the
            other possible estimates


        Output
        ------
        est_n_clusters, possibilities

        est_n_clusters: the estimated number of clustesr
        possibilities: local maxima of the given method
        """
        if method is None:
            method = self.gap_est_method

        est_n_clusters, possibilities = \
            estimate_n_clusters(cluster_sizes=self.cluster_sizes,
                                f=self.gap,
                                se=self.E_log_null_wcss_est_sd,
                                se_adj_factor=self.adj_factor,
                                method=method)

        self.gap_est_method = method
        self.est_n_clusters = est_n_clusters
        self.possible_n_clusters = possibilities

    def plot_wcss_curves(self):

        # plot observed log(WCSS)
        plt.plot(self.cluster_sizes,
                 self.log_obs_wcss,
                 marker="$O$",
                 color='blue',
                 ls='solid',
                 label='obs')

        # plot the expected log(WCSS) of the null references
        plt.plot(self.cluster_sizes,
                 self.E_log_null_wcss_est,
                 marker='$E$',
                 color='red',
                 ls='dashed',
                 label='E null')

        plt.xticks(self.cluster_sizes)
        plt.xlabel('number of clusters')
        plt.ylabel('log(WCSS)')

        plt.legend()

    def plot_gap(self, errorbars=True, include_est=True,
                 include_possibilities=False):

        if errorbars:
            # TODO: should we use s_adj for error bars?
            plt.errorbar(self.cluster_sizes,
                         self.gap,
                         color='black',
                         yerr=self.E_log_null_wcss_est_sd)

        else:
            plt.plot(self.cluster_sizes,
                     self.gap,
                     color='black',
                     marker='x')

        plt.xticks(self.cluster_sizes)
        plt.xlabel('number of clusters')
        plt.ylabel('gap')

        # maybe include the estimated numer of clusters
        if include_est:
            plt.axvline(x=self.est_n_clusters, color='red', label='estimate')

        # maybe include other possible estimates
        if include_possibilities:
            label = 'possibility'
            for n in self.possible_n_clusters:

                if n == self.est_n_clusters:
                    continue

                plt.axvline(x=n, color='blue', ls='dashed', lw=1, label=label)
                label = ''  # HACK: get only one 'possibility' label to show up

            plt.legend()

    def save(self, fname, compress=True, include_data=False,
             include_obs_cluster_labels=True):

        save_dict = {'ref_dist': self.ref_dist,
                     'B': self.B,
                     'cluster_sizes': self.cluster_sizes,
                     'gap_est_method': self.gap_est_method,
                     'clusterer_name': self.clusterer_name,
                     'clusterer_kwargs': self.clusterer_kwargs,
                     'obs_wcss':   self.obs_wcss,
                     'null_wcss_samples':   self.null_wcss_samples,
                     'est_n_clusters':   self.est_n_clusters,
                     'possible_n_clusters':   self.possible_n_clusters}

        if include_data:
            save_dict['X'] = self.X
            save_dict['U'] = self.U
            save_dict['D'] = self.D
            save_dict['V'] = self.V
        else:
            save_dict['X'] = None
            save_dict['U'] = None
            save_dict['D'] = None
            save_dict['V'] = None

        if include_obs_cluster_labels:
            save_dict['obs_cluster_labels'] = self.obs_cluster_labels
        else:
            save_dict['obs_cluster_labels'] = None

        joblib.dump(save_dict,
                    filename=fname,
                    compress=compress)

    @classmethod
    def load_from_dict(cls, load_dict):

        # initialize class
        GS = cls(clusterer=load_dict['clusterer_name'],
                 clusterer_kwargs=load_dict['clusterer_kwargs'],
                 cluster_sizes=load_dict['cluster_sizes'],
                 ref_dist=load_dict['ref_dist'],
                 B=load_dict['B'],
                 gap_est_method=load_dict['gap_est_method'])

        GS.obs_wcss = load_dict['obs_wcss']
        GS.null_wcss_samples = load_dict['null_wcss_samples']
        GS.est_n_clusters = load_dict['est_n_clusters']
        GS.possible_n_clusters = load_dict['possible_n_clusters']

        GS.X = load_dict['X']
        GS.U = load_dict['U']
        GS.D = load_dict['D']
        GS.V = load_dict['B']
        return GS

    @classmethod
    def load(cls, fname):
        load_dict = joblib.load(fname)

        return cls.load_from_dict(load_dict)

    @classmethod
    def from_precomputed_wcss(cls, cluster_sizes, obs_wcss,
                              null_wcss_samples, **kwargs):

        """
        Initializes GatStat object form precomputed obs_wcss and
        null_wcss_smaples.
        """

        assert len(obs_wcss) == len(cluster_sizes)
        assert null_wcss_samples.shape[0] == len(cluster_sizes)

        GS = cls(cluster_sizes=cluster_sizes, **kwargs)

        GS.obs_wcss = obs_wcss
        GS.null_wcss_samples = null_wcss_samples
        GS.B = null_wcss_samples.shape[1]  # NOTE: B may be differnt
        GS.compute_n_cluster_estimate()
        return GS
