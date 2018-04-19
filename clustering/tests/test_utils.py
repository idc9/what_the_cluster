from clustering.utils import svd_wrapper
import numpy as np

# TODO: turn this into nose tests

def main():

    X = np.random.normal(size=(20, 10))

    n = X.shape[0]
    d = X.shape[1]
    r = min(X.shape)

    U, D, V = svd_wrapper(X)

    assert U.shape == (n, r)
    assert len(D)== r
    assert V.shape == (d, r)


if __name__ == "__main__":
    main()
