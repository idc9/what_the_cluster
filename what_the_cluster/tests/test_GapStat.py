from clustering.GapStat import GapStat
from sklearn.datasets import make_blobs


# TODO: turn this into nose tests

def main():

    X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)

    gs = GapStat(ref_dist='uniform')
    est_n_clust = gs.estimate_n_clusters(X)
    assert est_n_clust == 3

    gs = GapStat(ref_dist='svd')
    est_n_clust = gs.estimate_n_clusters(X)
    assert est_n_clust == 3


def test_save_load():
    X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)

    gs = GapStat(ref_dist='svd')
    gs.compute_obs_clusters(X)
    gs.compute_svd_decomposition()

    fname = 'test'
    gs.save(fname)

    loaded = GapStat.load(fname)
    loaded.compute_obs_wcss()
    loaded.sample_ref_null_wcss()
    loaded.compute_n_cluster_estimate(method=gs.gap_est_method)

    assert loaded.est_n_clusters == 3


if __name__ == "__main__":
    main()
