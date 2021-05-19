import math
from utils import euclidean_dist

def _get_dist_matrix(data, eps):
    dist_matrix = [[True] * len(data) for i in range(len(data))]
    for p in range(len(data)):
        for q in range(p + 1, len(data)):
            dist = euclidean_dist(data[p], data[q])
            if (dist > eps):
                dist_matrix[p][q] = dist_matrix[q][p] = False
    return dist_matrix

def _range_query(p, dist_matrix):
    q_dists = dist_matrix[p]
    neighbors = [q for q, q_dist in enumerate(q_dists) if q_dist]
    return neighbors
    
def _expand_cluster(p, neighbors, labels, cluster, dist_matrix, min_pts):
    labels[p] = cluster
    while neighbors:
        q = neighbors.pop(0)
        if labels[q] == -1:
            labels[q] = cluster
            continue
        if labels[q] is not None:
            continue
        labels[q] = cluster
        new_neighbors = _range_query(q, dist_matrix)
        if len(new_neighbors) >= min_pts:
            neighbors = neighbors + new_neighbors

def dbscan(data, eps=0.5, min_pts=5):
    """
    Implementation of Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    
    Source:
        https://dl.acm.org/doi/10.1145/3068335
    Distance Measure:
        Euclidean distance
    Inputs:
        - "data":    A list of points to be clustered
        - "eps":     The neighborhood radius around a point (defaults to 0.5)
        - "min_pts": The minimum number of points required to form a dense region (defaults to 5)
    Outputs:
        A list of cluster labels, in the same order as the data. The -1 label corresponds to noise;
        otherwise clusters are numbered using consecutive integers starting from 0.
    """
    cluster = 0
    labels = [None] * len(data)
    dist_matrix = _get_dist_matrix(data, eps)
    for p in range(len(data)):
        if labels[p] is not None:
            continue
        neighbors = _range_query(p, dist_matrix)
        if len(neighbors) < min_pts:
            labels[p] = -1
            continue
        _expand_cluster(p, neighbors, labels, cluster, dist_matrix, min_pts)
        cluster += 1
    return labels
