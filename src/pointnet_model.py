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

class PointNet(nn.Module):
    def __init__(self, class_num=10):
        super(PointNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp_first = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp_second = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 200, 1),
            nn.BatchNorm1d(200),
            nn.ReLU()
        )
        self.mlp_third = nn.Sequential(
            nn.Conv1d(200, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, class_num, 1),
            nn.ReLU()
        )
        
        
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        #print(x.shape)
        x = self.input(x)
        x = self.mlp_first(x)
        #print(x.shape)
        x = self.mlp_second(x) 
        #print(x.shape)
        x = F.max_pool1d(x, 1000)
        #print(x.shape)
        x = self.mlp_third(x)
        #print(x.shape)
        x = x.squeeze(2)
        #print(x.shape)
        return F.log_softmax(x, dim=1)
    
