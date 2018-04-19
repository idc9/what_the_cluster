what_the_cluster
----

**author**: `Iain Carmichael`_

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
iain@unc.edu.

Overview
========

This package contains some algoritms to estimate the number of clusters for an arbitrary clustering algorithm.  While many of these can be found elsewhere (see below) for whatever reason I found it useful to them implement differently in a way which is more useful for what I was doing. Hopefully you find these useful as well.

Currently implements

- `the gap statistic`_

Installation
============
This is currently an informal package under development so I've only made it installable from github.

::

    git clone https://github.com/idc9/clustering.git
    python setup.py install

Example
=======

.. code:: python

    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # sample a toy 2-dim example with 3 clusters
    X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # estimate number of clusters with the gap statistic
    from clustering.GapStat import GapStat
    gs = GapStat(clusterer='kmeans',  # an arbitrary clustering algorithm can be provided
             cluster_sizes=range(1, 11),
             ref_dist='uniform',  # either 'uniform' or 'svd'
             B=10)  # number of samples from the reference null distribution

    gs.estimate_n_clusters(X)
    gs.plot_wcss_curves()
    gs.plot_gap()

For some more example code see `these example notebooks`_.

Help and Support
================

Additional documentation, examples and code revisions are coming soon.
For questions, issues or feature requests please reach out to Iain:
iain@unc.edu.

Documentation
^^^^^^^^^^^^^

The source code is located on github:
`https://github.com/idc9/clustering`_.

Testing
^^^^^^^

Testing is done using `nose`_.

Contributing
^^^^^^^^^^^^

We welcome contributions to make this a stronger package: data examples,
bug fixes, spelling errors, new features, etc.


Other Clustering Packages
^^^^^^^^^^^^^^^^^^^^^^^^^
There are many other clustering packages out there including. This package overlaps a little bit with some of the below, basic python implementations, but does things the way I needed them done for my project. For example, the user can supply an aribtrary clustering algorithm implmented in python to this package.

In python

- sklearn's `clustering functionality`_

- https://github.com/milesgranger/gap_statistic

- https://github.com/minddrummer/gap

- https://gist.github.com/michiexile/5635273

- https://github.com/Zelazny7/gap-statistic

- https://github.com/annoviko/pyclustering

- https://github.com/topics/clustering-algorithm?l=python

In R

- https://cran.r-project.org/web/packages/cluster/cluster.pdf

- http://had.co.nz/clusterfly/

- https://github.com/nolanlab/Rclusterpp

- https://github.com/bwrc/corecluster-r

- http://danifold.net/fastcluster.html


.. _Iain Carmichael: https://idc9.github.io/
.. _the gap statistic: https://web.stanford.edu/~hastie/Papers/gap.pdf
.. _these example notebooks: https://github.com/idc9/what_the_cluster/tree/master/doc
.. _`https://github.com/idc9/what_the_cluster`: https://github.com/idc9/what_the_cluster
.. _clustering functionality: http://scikit-learn.org/stable/modules/clustering.html
