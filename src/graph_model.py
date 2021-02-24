import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, ASAPooling


class GCN(nn.Module):
    def __init__(self, pool='SAG', ratio=0.5, class_num=10):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 16, add_self_loops = True, normalize = True)
        self.conv2 = GCNConv(16, 32, add_self_loops = True, normalize = True)
        self.conv3 = GCNConv(32, 64, add_self_loops = True, normalize = True)
        if pool == 'SAG':
            self.pool1 = SAGPooling(in_channels=32, ratio=ratio)
            self.pool2 = SAGPooling(in_channels=64, ratio=ratio)
        else:
            self.pool1 = ASAPooling(in_channels=32, ratio=ratio)
            self.pool2 = ASAPooling(in_channels=64, ratio=ratio)
        self.conv4 = GCNConv(64, 32, add_self_loops = True, normalize = True)
        self.conv5 = GCNConv(32, 32, add_self_loops = True, normalize = True)
        self.conv6 = GCNConv(32, class_num, add_self_loops = True, normalize = True)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.view(-1, 3)
        edge_index = edge_index.view(2, -1)
        edge_attr = edge_attr.view(-1)
        #print(edge_index.shape, edge_index.type)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        temp = self.pool1(x, edge_index, edge_attr)
        x, edge_index, edge_attr = temp[0], temp[1], temp[2]
        
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        temp = self.pool2(x, edge_index, edge_attr)
        x, edge_index, edge_attr = temp[0], temp[1], temp[2]
        
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        
        x = self.conv5(x, edge_index, edge_attr)
        x = F.relu(x)
        
        x = self.conv6(x, edge_index, edge_attr)
        x = F.relu(x)
        
        x = torch.max(x, dim=0, keepdim=True)[0]
        return F.log_softmax(x, dim=1)