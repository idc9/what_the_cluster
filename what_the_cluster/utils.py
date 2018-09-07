from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd
import numpy as np


def _is_strictly_increasing(a):
    """
    Checks if an array is strictly increasing
    """
    return all(a[i] < a[i+1] for i in range(len(a) - 1))


def _count_none(*args):
    """
    Counts how many of the arguments are None
    """
    return sum([a is None for a in args])


def svd_wrapper(X, rank=None):
    """
    Computes the (possibly partial) SVD of a matrix.

    Parameters
    ----------
    X: either dense or sparse
    rank: rank of the desired SVD (required for sparse matrices)

    Output
    ------
    U, D, V
    the columns of U are the left singular vectors
    the COLUMNS of V are the left singular vectors

    """

    if rank is not None and rank > min(X.shape):
        raise ValueError('rank must be <= the smallest dimension of X. rank= {} was passed in while X.shape = {}'.format(rank, X.shape))

    if rank is None or rank == min(X.shape):
        U, D, V = full_svd(X, full_matrices=False)
        V = V.T
    else:
        scipy_svds = svds(X, rank)
        U, D, V = fix_scipy_svds(scipy_svds)

    return U, D, V


def fix_scipy_svds(scipy_svds):
    """
    scipy.sparse.linalg.svds orders the singular values backwards,
    this function fixes this insanity and returns the singular values
    in decreasing order

    Parameters
    ----------
    scipy_svds: the out put from scipy.sparse.linalg.svds

    Output
    ------
    U, D, V
    ordered in decreasing singular values
    """
    U, D, V = scipy_svds

    sv_reordering = np.argsort(-D)

    U = U[:, sv_reordering]
    D = D[sv_reordering]
    V = V.T[:, sv_reordering]

    return U, D, V
