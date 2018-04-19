from scipy.linalg import svd


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


def svd_wrapper(X):
    """
    Computes the full SVD of a matrix X (must be dense). Returns
    scores (U), singular values (D) and loadings (V) ordered by decreasing
    singular value.

    Loadings are returned such that the loadings vectors are
    the columns of V. I.e. if X is a (n x d) matrix then
    V will be a (d x r) matrix where r = min(n, d).
    """
    U, D, V = svd(X, full_matrices=False)
    V = V.T
    return U, D, V
