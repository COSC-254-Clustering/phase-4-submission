import math
from utils import haversine_dist

def _get_neighbors(data, p, eps, ceps):
    neighbors = []
    p_coords = (data[p][0], data[p][1])
    for q in range(p, len(data)):
        q_coords = (data[q][0], data[q][1])
        dist = haversine_dist(p_coords, q_coords)
        if dist <= eps:
            neighbors.append(q)
        elif dist > ceps:
            break
    return neighbors

def _expand_cluster(data, p, neighbors, labels, cluster_id, eps, ceps, min_pts, max_id):
    cluster = []
    while neighbors:
        q = neighbors.pop(0)
        if q > max_id:
            max_id = q
        if labels[q] == -1:
            labels[q] = cluster_id
            cluster.append(q)
            continue
        if labels[q] is not None:
            continue
        labels[q] = cluster_id
        cluster.append(q)
        new_neighbors = _get_neighbors(data, q, eps, ceps)
        if len(new_neighbors) >= min_pts:
            neighbors = neighbors + new_neighbors
    return cluster, max_id

def _clean_labels(labels):
    for i, label in enumerate(labels):
        if label is None:
            labels[i] = labels[i - 1]

def t_dbscan(data, eps=10, ceps=50, min_pts=6):
    """
    Implementation of T-DBSCAN Clustering for Detecting Stops in GPS Trajectories
    
    Source:
        http://dx.doi.org/10.3991/ijoe.v10i6.3881
    Inputs:
        - "data":    A list of GPS coordinates (as [lat, lon] lists)
        - "eps":     The inner radius for the density area of a point (defaults to 10 meters)
        - "ceps":    The outer radius for limiting the density search range (defaults to 50 meters)
        - "min_pts": The minimum number of points required to form a dense region (defaults to 6)
    Outputs:
        A list of cluster labels, in the same order as the data. The -1 label corresponds to
        noise; otherwise clusters are numbered using consecutive integers starting from 1.
    """
    cluster_id = 0
    clusters = {}
    max_id = -1
    labels = [None] * len(data)
    for p in range(len(data)):
        if p <= max_id:
            continue
        neighbors = _get_neighbors(data, p, eps, ceps)
        max_id = p
        if len(neighbors) < min_pts:
            labels[p] = -1
            continue
        cluster_id += 1
        cluster, max_id = _expand_cluster(data, p, neighbors, labels, cluster_id, eps, ceps, min_pts, max_id)
        clusters[cluster_id] = clusters.get(cluster_id, []) + cluster
    _clean_labels(labels)
    return labels