from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from scipy.sparse import issparse

from what_the_cluster.gapstat_utils import get_pooled_wcss, estimate_n_clusters
from what_the_cluster.reference_dists import sample_svd_null, sample_uniform_null
from what_the_cluster.utils import _is_strictly_increasing, _count_none, svd_wrapper
from what_the_cluster.clusterers import get_clusterer

# TODO: implement seeds
# TODO: give clusterer the option to return additional data
# TODO: give user the ability to input pre-sampled reference distributions
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
        # self.X = None  # observed data
        # self.U = None  # U, D, V are SVD of X
        # self.D = None
        # self.V = None

        # self.obs_cluster_labels = None
        # self.obs_wcss = None
        # self.null_wcss_samples = None
        # self.est_n_clusters = None
        # self.possible_n_clusters = None

        # self.metadata = {}

    def get_params(self):
        return {'clusterer': self.clusterer,
                'clusterer_kwargs': self.clusterer_kwargs,
                'cluster_sizes': self.cluster_sizes,
                'ref_dist': self.ref_dist,
                'B': self.B,
                'gap_est_method': self.gap_est_method}

    def fit(self, X, cluster_labels=None,
            U=None, D=None, V=None):
        """
        Estimates the number of clusters using the gap statistic.

        Parameters
        ----------
        X (matrix): the observed data with observations on the rows.

        cluster_labels (None or matrix, observations x len(cluster_sizes)): matrix
            containing the observed cluster labels on the columns for each
            value of n_clusters.

            If None then will uses clusterer to estimate the number of clusters
            using the provided clusterer


        U, D, V: the precomputed SVD of X see set_svd_decomposition() for
            details. These are only used if ref_dist = 'svd'. If they are not
            provided then will compute them.
        """
        if type(X) == pd.DataFrame:
            self.var_names = np.array(X.columns)
        else:
            self.var_names = np.array(range(X.shape[1]))

        if not issparse(X):
            X = np.array(X)

        if cluster_labels is None:
            cluster_labels = self.compute_obs_clusters(X)
            assert cluster_labels.shape == (X.shape[0], len(self.cluster_sizes))

        if self.ref_dist == 'svd':
            if _count_none(U, D, V) == 3:
                U, D, V = svd_wrapper(X)

            elif _count_none(U, D, V) != 0:
                raise ValueError('U, D, V must all be provided or be set to None')

        self.obs_wcss = self.compute_obs_wcss(X, cluster_labels)
        self.null_wcss_samples = self.sample_ref_null_wcss(X, U=U, D=D, V=V)
        self.compute_n_cluster_estimate(method=self.gap_est_method)

        return self

    @property
    def est_cluster_memberships(self):
        """
        Returns the estimated cluster memberships
        """
        assert self.est_n_clusters is not None
        est_cluster_size_ind = np.where(
            np.array(self.cluster_sizes) == self.est_n_clusters)[0][0]
        return self.obs_cluster_labels[:, est_cluster_size_ind]

    def compute_obs_clusters(self, X):

        obs_cluster_labels = np.zeros((X.shape[0], len(self.cluster_sizes)))

        for i, n_clusters in enumerate(self.cluster_sizes):
            obs_cluster_labels[:, i] = self.clusterer(X, n_clusters)

        return obs_cluster_labels

    def compute_obs_wcss(self, X, obs_cluster_labels):
        """
        Computes the within class sum of squres for the observed clusters.
        """
        n_cluster_sizes = len(self.cluster_sizes)
        obs_wcss = np.zeros(n_cluster_sizes)

        for j in range(n_cluster_sizes):
            # make sure the number of unique cluster labels is equal to
            # the preported number of clusters
            # TODO: we might not want this restrictin
            assert len(set(obs_cluster_labels[:, j])) \
                == self.cluster_sizes[j]

            obs_wcss[j] = get_pooled_wcss(X, obs_cluster_labels[:, j])

        return obs_wcss

    def sample_null_reference(self, X, U=None, D=None, V=None):

        if self.ref_dist == 'uniform':
            return sample_uniform_null(X)
        elif self.ref_dist == 'svd':
            return sample_svd_null(X, U, D, V)

    def sample_ref_null_wcss(self, X, U=None, D=None, V=None):

        null_wcss_samples = np.zeros((len(self.cluster_sizes), self.B))

        for b in range(self.B):
            # sample null reference distribution
            X_null = self.sample_null_reference(X, U=U, D=D, V=V)

            # cluster X_null for the specified n_clusters
            for i, n_clusters in enumerate(self.cluster_sizes):
                # cluster. null sample
                null_cluster_labels = self.clusterer(X_null, n_clusters)

                null_wcss_samples[i, b] = get_pooled_wcss(X_null,
                                                          null_cluster_labels)

        return null_wcss_samples

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
            plt.axvline(x=self.est_n_clusters, color='red',
                        label='estimated {} clusters'.
                        format(self.est_n_clusters))

        # maybe include other possible estimates
        if include_possibilities:
            label = 'possibility'
            for n in self.possible_n_clusters:

                if n == self.est_n_clusters:
                    continue

                plt.axvline(x=n, color='blue', ls='dashed', lw=1, label=label)
                label = ''  # HACK: get only one 'possibility' label to show up

        plt.legend()

    def save(self, fname, compress=True, include_data=False):

        # save_dict = {'ref_dist': self.ref_dist,
        #              'B': self.B,
        #              'cluster_sizes': self.cluster_sizes,
        #              'gap_est_method': self.gap_est_method,
        #              'clusterer_name': self.clusterer_name,
        #              'clusterer_kwargs': self.clusterer_kwargs,
        #              'obs_cluster_labels': self.obs_cluster_labels,
        #              'obs_wcss':   self.obs_wcss,
        #              'null_wcss_samples':   self.null_wcss_samples,
        #              'est_n_clusters':   self.est_n_clusters,
        #              'possible_n_clusters':   self.possible_n_clusters,
        #              'metadata': self.metadata}

        # if include_data:
        #     save_dict['X'] = self.X
        #     save_dict['U'] = self.U
        #     save_dict['D'] = self.D
        #     save_dict['V'] = self.V
        # else:
        #     save_dict['X'] = None
        #     save_dict['U'] = None
        #     save_dict['D'] = None
        #     save_dict['V'] = None

        joblib.dump(self,
                    filename=fname,
                    compress=compress)

    # @classmethod
    # def load_from_dict(cls, load_dict):

    #     # initialize class
    #     GS = cls(clusterer=load_dict['clusterer_name'],
    #              clusterer_kwargs=load_dict['clusterer_kwargs'],
    #              cluster_sizes=load_dict['cluster_sizes'],
    #              ref_dist=load_dict['ref_dist'],
    #              B=load_dict['B'],
    #              gap_est_method=load_dict['gap_est_method'])

    #     GS.obs_cluster_labels = load_dict['obs_cluster_labels']

    #     GS.obs_wcss = load_dict['obs_wcss']
    #     GS.null_wcss_samples = load_dict['null_wcss_samples']
    #     GS.est_n_clusters = load_dict['est_n_clusters']
    #     GS.possible_n_clusters = load_dict['possible_n_clusters']

    #     GS.X = load_dict['X']
    #     GS.U = load_dict['U']
    #     GS.D = load_dict['D']
    #     GS.V = load_dict['B']

    #     GS.metadata = load_dict['metadata']
    #     return GS

    @classmethod
    def load(cls, fname):
        # load_dict = joblib.load(fname)
        # return cls.load_from_dict(load_dict)
        return joblib.load(fname)

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
