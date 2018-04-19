"""
These functions are is a higher order functions i.e. they returns a function.
In particulary, they will return a function, clusterer(), which takes two
arguments (the data and the desired number of clusters) and returns the
estimated cluster labels for each observation.

Note the **kwargs allow the user to specify additional arguments. See the
original documentation for each function for a list of available arguments.

"""

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def get_clusterer(clusterer, clusterer_kwargs):
    if clusterer == 'kmeans':
        return kmeans_clusterer(**clusterer_kwargs)

    else:
        raise ValueError('%s is not a valid clusterer' % clusterer)


def kmeans_clusterer(**kwargs):
    """
    Returns a function which runs K-Means

    For documentation see http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    def clusterer(X, n_clusters):
        """
        Parameters
        ----------
        X (matrix): the dataset to cluster
        n_clusters (int): number of clusters to find
        """
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
        return clusterer.fit_predict(X)

    return clusterer


def spectral_clusterer(**kwargs):
    """
    Returns a function which runs spectral clustering

    For documentation see http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
    """
    def clusterer(X, n_clusters):
        """
        Parameters
        ----------
        X (matrix): the dataset to cluster
        n_clusters (int): number of clusters to find
        """
        clusterer = SpectralClustering(n_clusters=n_clusters, **kwargs)
        return clusterer.fit_predict(X)

    return clusterer


def agglo_clusterer(**kwargs):
    """
    Returns a function which runs agglomerative clustering

    For documentation see http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    """
    def clusterer(X, n_clusters):
        """
        Parameters
        ----------
        X (matrix): the dataset to cluster
        n_clusters (int): number of clusters to find
        """
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        return clusterer.fit_predict(X)

    return clusterer


# TODO: figure this one out
def gmm_clusterer(**kwargs):
    """
    Returns a function which runs clustering using a guassian mixture model

    For documentation see http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """
    def clusterer(X, n_clusters):
        """
        Parameters
        ----------
        X (matrix): the dataset to cluster
        n_clusters (int): number of clusters to find
        """
        clusterer = GaussianMixture(n_components=n_clusters, **kwargs)
        raise NotImplementedError
    return clusterer


# TODO: figure this one out
def bayes_gmm_clusterer(**kwargs):
    """
    Returns a function which runs clustering using a guassian mixture model
    fit using variational bayes

    For documentation see http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
    """
    def clusterer(X, n_clusters):
        """
        Parameters
        ----------
        X (matrix): the dataset to cluster
        n_clusters (int): number of clusters to find
        """
        clusterer = BayesianGaussianMixture(n_components=n_clusters, **kwargs)
        raise NotImplementedError
    return clusterer
