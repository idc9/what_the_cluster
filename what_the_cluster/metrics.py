import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, \
    normalized_mutual_info_score, v_measure_score, \
    adjusted_rand_score


def compute_cluster_metrics(A, B):
    """
    Compares two sets of cluster assignments
    Note each of these metrics are symmetric i.e. A,B is the same as B, A
    For more details see http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    """
    assert len(A) == len(B)

    metrics = {}
    metrics['adj_rank'] = adjusted_rand_score(A, B)
    metrics['AMI'] = adjusted_mutual_info_score(A, B)
    metrics['NMI'] = normalized_mutual_info_score(A, B)
    metrics['v_measure'] = v_measure_score(A, B)
    return pd.Series(metrics)
