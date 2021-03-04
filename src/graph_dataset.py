import torch
import torch.utils.data as data
from sklearn import preprocessing
import h5py
import numpy as np
class GCNdata(data.Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(labels)
        self.dct = le.classes_
        
    def __getitem__(self, index):
        f = h5py.File(self.paths[index], 'r')
        edge_w = f['edge_weight'][:]
        edges = f['edges'][:]
        nodes = f['nodes'][:]
        edges = torch.tensor(edges, dtype = torch.long)
        x = torch.tensor(nodes, dtype=torch.float)
        weights = torch.tensor(edge_w, dtype=torch.float)
        f.close()
        y = torch.from_numpy(np.array(self.labels[index]))
        return x, edges, weights, y
    
    def __len__(self):
        return len(self.paths)
    
    def getdct(self):
        return self.dct