import numpy as np


def sample_uniform_null(X):
    """
    Samples the uniform null reference distribution. TODO: explain more
    """
    ranges = variable_ranges(X)
    X_null = np.random.random_sample(X.shape)
    X_null = (ranges[1, :] - ranges[0, :]) * X_null + ranges[0, :]
    return X_null


def sample_svd_null(X, U, D, V):
    """
    Samples the SVD null reference distribution. TODO: explain more
    """
    # un normalized scores i.e. UD = XV
    un_scores = U * D

    X_null_scores_space = sample_uniform_null(un_scores)
    return X_null_scores_space.dot(V.T)


def variable_ranges(X):
    """
    Computest the ranges for variables of X (columns)
    """
    return np.apply_along_axis(lambda c: (min(c), max(c)), axis=0, arr=X)
