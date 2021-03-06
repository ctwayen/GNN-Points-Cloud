from src import graph_construct, visual
import h5py
import networkx as nx
import json
import argparse
import torch
import os
import pandas as pd
import numpy as np
import shutil
import glob
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import os.path as osp

def download(root='data/modelnet/', name='40'):
    urls = {
        '10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }
    os.makedirs(root, exist_ok=True)
    path = download_url(urls[name], root)
    extract_zip(path, root)
    os.unlink(path)
    folder = osp.join(root, 'ModelNet{}'.format(name))
    # Delete osx metadata generated during compression of ModelNet10
    metadata_folder = osp.join(root, '__MACOSX')
    if osp.exists(metadata_folder):
        shutil.rmtree(metadata_folder)
        
        
def extract(path):
    f = open(path, 'r')
    lines = f.readlines()
    if lines[0] == 'OFF\n':
        num = int(float(lines[1].split(" ")[0]))
        pts = []
        for i in range(2, 2+num):
            temp = lines[i][:-1].split(' ')
            pts.append([float(temp[0]), float(temp[1]), float(temp[2])])
    else:
        num = int(float(lines[0].split(" ")[0][3:]))
        pts = []
        for i in range(1, 1+num):
            temp = lines[i][:-1].split(' ')
            pts.append([float(temp[0]), float(temp[1]), float(temp[2])])
    return pts

def create_pt(base = 'data/modelnet/ModelNet40/', output_base = 'data/modelnet/modelnet_points/'):
    paths = {}
    for obj in os.listdir(base):
        if obj != '.DS_Store' and obj != 'README.txt':
            obj_base_train = base + obj + '/train/'
            obj_base_test = base + obj + '/test/'
            paths[obj] = []
            for sample in os.listdir(obj_base_train):
                paths[obj].append(obj_base_train + sample)
            for sample in os.listdir(obj_base_test):
                paths[obj].append(obj_base_test + sample)
                
    os.makedirs(output_base, exist_ok=True)

    for obj in paths.keys():
        h5f = h5py.File(output_base + str(obj)+'.h5', 'w')
        for path, i in zip(paths[obj], range(len(paths[obj]))):
            if path[-3:] == 'off':
                temp = np.array(extract(path))
                h5f.create_dataset('object_' + str(i), data=temp)
        h5f.close()
        
def construct_graph_knn(pts_num = 1000, base = 'data/modelnet/modelnet_points/', k=15):
    output_base = 'data/modelnet/modelnet_graph_k{x}'.format(x=k)
    print('Output will stored in ' + output_base)
    print('You will need this path for later on model training')
    os.makedirs(output_base, exist_ok=True)
    for obj in os.listdir(base):
        cat = obj[:-3]
        print(obj)
        if obj[-2:] == 'h5':
            os.makedirs(output_base + '/' + cat, exist_ok=True)
            f = h5py.File(base + obj, 'r')
            for key in f.keys():
                if f[key][:].shape[0] >= pts_num:
                    pts = graph_construct.pts_norm(graph_construct.pts_sample(f[key][:], pts_num))
                    if np.isnan(pts).any():
                        continue
                    temp = graph_construct.graph_construct_kneigh(pts, k=k)
                    filename = output_base + '/' + cat + '/' + key + '.h5'
                    out = h5py.File(filename, 'w')
                    out.create_dataset('edges', data=temp[0])
                    out.create_dataset('edge_weight', data=temp[1])
                    out.create_dataset('nodes', data=pts)
                    out.close()
        
def construct_graph_radius(pts_num = 1000, base = 'data/modelnet/modelnet_points/', radius=0.1):
    output_base = 'data/modelnet/modelnet_graph_r{x}/'.format(x=radius)
    print('Output will stored in ' + output_base)
    print('You will need this path for later on model training')
    os.makedirs(output_base, exist_ok=True)
    for obj in os.listdir(base):
        cat = obj[:-3]
        print(obj)
        if obj[-2:] == 'h5':
            os.makedirs(output_base + '/' + cat, exist_ok=True)
            f = h5py.File(base + obj, 'r')
            for key in f.keys():
                if f[key][:].shape[0] >= pts_num:
                    pts = graph_construct.pts_norm(graph_construct.pts_sample(f[key][:], pts_num))
                    if np.isnan(pts).any():
                        continue
                    temp = graph_construct.graph_construct_radius(pts, r=radius)
                    filename = output_base + '/' + cat + '/' + key + '.h5'
                    out = h5py.File(filename, 'w')
                    out.create_dataset('edges', data=temp[0])
                    out.create_dataset('edge_weight', data=temp[1])
                    out.create_dataset('nodes', data=pts)
                    out.close()