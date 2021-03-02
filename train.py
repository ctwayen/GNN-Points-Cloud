import torch
import os
from src import graph_trainer
from src import graph_dataset
from src import graph_model
import json
import pandas as pd
import os
from src import graph_construct
import os
import numpy as np
import h5py

def construct_graph_radius(pts_num = 1000, base = 'data/modelnet/modelnet_points/', radius=0.1):
    output_base = 'data/modelnet/modelnet_graph_r{x}/'.format(x=radius)
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
class run():
    def __init__(self, args):
        paths = []
        labels = []
        base = args['base']
        for obj in os.listdir(base):
            temp = base + obj
            if args['data'] == '10':
                if obj in ['sofa', 'airplane', 'vase', 'chair', 'toilet', 'bookshelf', 'bed', 'monitor', 'piano', 'bottle']:
                    for file in os.listdir(temp):
                        paths.append(temp+'/' + file)
                        labels.append(obj)
            else:
                for file in os.listdir(temp):
                        paths.append(temp+'/' + file)
                        labels.append(obj)
        dataset = graph_dataset.GCNdata(paths, labels)
        total = len(dataset)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [total - int(total * args['val_size']),int(total * args['val_size'])])
        print(total, len(train_dataset), len(test_dataset))
        if args['data'] == '10':
            class_num = 10
        else:
            class_num = 40
        if args['model'] == 'GCN':
            self.model = graph_model.GCN(pool=args['pool'], ratio=args['ratio'], class_num=class_num)
            opts = {
                    'lr': args['lr'],
                    'epochs': args['epoch'],
                    'batch_size': args['bs'],
                    'model_path': args['model_path']
            }
        self.Train = graph_trainer.trainer(model = self.model,
                      train_set = train_dataset,
                      test_set = test_dataset,opts = opts)
    def process(self):
        self.Train.train()
        return self.Train.get_stats()
    
def main():
    base = 'data/modelnet/'
    for obj in os.listdir(base):
        if obj != 'ModelNet40' and obj != '.ipynb_checkpoints' and obj != 'modelnet_points':
            print(base + obj)
            temp = {
                'data': '10',
                'base': base + obj + '/',
                'model': 'GCN',
                'pool': 'SAG',
                'ratio': 0.4,
                'val_size': 0.2,
                'lr': 5e-4,
                'epoch': 30,
                'bs': 32,
                'model_path': 'trained_models/' + obj + '.pt'
            }
            temp = run(temp)
            temp_out = 'config/model_results/' + obj + '.csv'
            print(temp_out)
            test = temp.process()
            print(test)
            out = pd.DataFrame()
            out['epoch'] = [x[0] for x in test]
            out['train_ls'] = [x[1] for x in test]
            out['test_ls'] = [x[2] for x in test]
            out['test_acc'] = [x[3] for x in test]
            out.to_csv(temp_out, index=False)
if __name__ == '__main__':
    main()