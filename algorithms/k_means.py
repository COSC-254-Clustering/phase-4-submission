import random
import statistics
from utils import euclidean_dist

def _init_centers(data, k, seed):
    random.seed(seed)
    init_centers = random.sample(data, k)
    return init_centers

def _get_dists(data, centers):
    dists = [None] * len(data)
    for p in range(len(data)):
        dists[p] = [euclidean_dist(data[p], center) for center in centers]
    return dists

def _get_labels(data, dists):
    labels = [None] * len(data)
    for p in range(len(data)):
        p_dists = dists[p]
        labels[p] = p_dists.index(min(p_dists))
    return labels

def _recompute_centers(data, k, labels):
    centers = [None] * k
    for i in range(k):
        cluster_pts = [data[p] for p in range(len(data)) if labels[p] == i]
        coord_groups = zip(*cluster_pts)
        centers[i] = [statistics.mean(coord_group) for coord_group in coord_groups]
    return centers

def _no_shift(centers, new_centers, tol):
    shifts = [euclidean_dist(centers[i], new_centers[i]) for i in range(len(centers))]
    no_shift = all([shift <= tol for shift in shifts])
    return no_shift

def k_means(data, k, tol=0.0001, max_iter=100, seed=1):
    """
    Implementation of k-Means Clustering

    Inputs:
        - "data":     A list of points to be clustered
        - "k":        The number of clusters to identify
        - "tol":      The tolerance for no further shift in the centers (defaults to 0.0001)
        - "max_iter": The maximum number of iterations (defaults to 100)
        - "seed":     The seed for the random initialization of centers (defaults to 1)
    Outputs:
        A list of cluster labels, in the same order as the data. The clusters are numbered
        using consecutive integers starting from 0.
    """
    centers = _init_centers(data, k, seed)
    for i in range(max_iter):
        dists = _get_dists(data, centers)
        labels = _get_labels(data, dists)
        new_centers = _recompute_centers(data, k, labels)
        if _no_shift(centers, new_centers, tol):
            break
        centers = new_centers
    return labels
