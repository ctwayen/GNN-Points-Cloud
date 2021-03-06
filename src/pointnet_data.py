import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, SAGPooling, ASAPooling, EdgePooling
from torch_geometric.data import Data
import os
import h5py
import torch.utils.data as data
from sklearn import preprocessing
import numpy as np

class Pointdata(data.Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(labels)
    def __getitem__(self, index):
        f = h5py.File(self.paths[index], 'r')
        nodes = f['nodes'][:]
        x = torch.tensor(nodes, dtype=torch.float)
        f.close()
        y = torch.from_numpy(np.array(self.labels[index]))
        return x, y
    
    def __len__(self):
        return len(self.paths)