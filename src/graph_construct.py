import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def pts_sample(pts, n_pts):
    idx = np.random.choice(pts.shape[0], n_pts, replace=False)
    return pts[idx,:]

def pts_norm(pts):
    pts = pts - pts.min(axis=0)
    return pts/pts.max(axis=0)

def graph_construct_kneigh(pts, k=30):
    nbrs = NearestNeighbors(algorithm='auto', leaf_size=30, p=2, n_neighbors=k).fit(pts)
    out = nbrs.kneighbors_graph(pts, mode='distance').todense()
    G=nx.from_numpy_matrix(out, create_using=nx.DiGraph)
    edges = [[x[0] for x in G.edges()], [x[1] for x in G.edges]]
    weights = [x[2]['weight'] for x in G.edges(data=True)]
    return np.array(edges), np.array(weights)

def graph_construct_full(pts):
    dist = DistanceMetric.get_metric('euclidean')
    return dist.pairwise(pts)

def graph_construct_radius(pts, r):
    nbrs = NearestNeighbors(algorithm='auto', leaf_size=30, p=2,
         radius=r).fit(pts)
    out = nbrs.radius_neighbors_graph(pts, mode='distance').todense()
    G=nx.from_numpy_matrix(out, create_using=nx.DiGraph)
    edges = [[x[0] for x in G.edges()], [x[1] for x in G.edges]]
    weights = [x[2]['weight'] for x in G.edges(data=True)]
    return np.array(edges), np.array(weights)