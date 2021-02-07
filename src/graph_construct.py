import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors

def pts_sample(pts, n_pts):
    idx = np.random.choice(pts.shape[0], n_pts, replace=False)
    return pts[idx,:]

def pts_norm(pts):
    pts = pts - pts.min(axis=0)
    return pts/pts.max(axis=0)

def graph_construct_kneigh(pts, k):
    nbrs = NearestNeighbors(algorithm='auto', leaf_size=30, p=2, n_neighbors=10,
             radius=0.1).fit(pts)
    out = nbrs.kneighbors_graph(pts, mode='distance').todense()
    return out

def graph_construct_full(pts):
    dist = DistanceMetric.get_metric('euclidean')
    return dist.pairwise(pts)

def graph_construct_radius(pts, r):
    nbrs = NearestNeighbors(algorithm='auto', leaf_size=30, p=2, n_neighbors=10,
         radius=0.1).fit(pts)
    out = nbrs.radius_neighbors_graph(pts, mode='distance').todense()
    return out