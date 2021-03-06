from src import graph_construct, visual, graph_dataset, graph_model, graph_trainer, point_train, pointnet_data, pointnet_model, create
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
        if args['model'] == 'GCN':
            dataset = graph_dataset.GCNdata(paths, labels)
            total = len(dataset)
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [total - int(total * args['val_size']),int(total * args['val_size'])])
            print('Total, training size, and test size')
            print(total, len(train_dataset), len(test_dataset))
            if args['data'] == '10':
                class_num = 10
            else:
                class_num = 40
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
        else:
            dataset = pointnet_data.Pointdata(paths, labels)
            total = len(dataset)
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [total - int(total * args['val_size']),int(total * args['val_size'])])
            print('Total, training size, and test size')
            print(total, len(train_dataset), len(test_dataset))
            if args['data'] == '10':
                class_num = 10
            else:
                class_num = 40
            self.model = pointnet_model.PointNet(class_num=class_num)
            opts = {
                    'lr': args['lr'],
                    'epochs': args['epoch'],
                    'batch_size': args['bs'],
                    'model_path': args['model_path']
            }
            self.Train = point_train.Trainer(model = self.model,
                          train_set = train_dataset,
                          test_set = test_dataset,opts = opts)
    def process(self):
        self.Train.train()
        return self.Train.get_stats()
    
def main():
    
    ## Basic
    parser = argparse.ArgumentParser(description='GNN-Points-Cloud')
    parser.add_argument('testorall', help='running test')
    parser.add_argument('--mode', type=str, default='train', choices=['download', 'train'],
                        help='Download or train')
    
    ### Download

    parser.add_argument('--method', type=str, default='knn', choices=['knn', 'fix_radius'],
                        help='graph contruct method to use')
    parser.add_argument('--k', type=int, default=30, help='parameter k')
    parser.add_argument('--r', type=float, default=0.1, help='parameter r')
    

    
    # train
    parser.add_argument('--model', type=str, default='GCN', choices=['pointNet', 'GCN'],
                        help='Model to use. Support GCN and PointNet')
    parser.add_argument('--lr', type=float, default=5e-4, help='parameter learning rate')
    parser.add_argument('--bs', type=int, default=32, help='parameter batch size')
    parser.add_argument('--ratio', type=float, default=0.4, help='parameter pooling ratio')
    parser.add_argument('--base', type=str, default='data/modelnet/modelnet_graph_k30/', help='dataset path')
    parser.add_argument('--data', type=str, default='10', choices=['10', '40'], help='Running on 10 or 40 classes')
    parser.add_argument('--epoch', type=int, default=30, help='parameter epoch')
    parser.add_argument('--val_size', type=float, default=0.2, help='parameter validation size ratio. E.g: 0.2')
    parser.add_argument('--model_path', type=str, default='trained_models/user_result/modelnet_graph_k30.pt')
    parser.add_argument('--output_path', type=str, default='config/model_results/user_result/modelnet_graph_k30.csv')
    parser.add_argument('--pool', type=str, default='SAG', choices=['SAG', 'ASA'])
    
    # Test
    parser.add_argument('--graph_image_path', type=str, default=None,
                        help='The path to store the output graph_visual_image_path')
    parser.add_argument('--points_image_path', type=str, default=None,
                        help='The path to store the output points_visual_image_path')
    parser.add_argument('--visual_base', type=str, default='x', choices=['x', 'y', 'z'],
                        help='The base axis to visual graph and points')
    
    
    args = parser.parse_args()
    print('-----------------------------------------')
    os.makedirs('data/modelnet/', exist_ok=True)
    if args.testorall == 'test':
        print('------------------------------------------------------')
        print("Running test will read a test points cloud data, consturct graph based on it, and then visualize it.")
        print("You could specify the path for the generated image with argument graph_image_path and points_image_path")
        print("If no path speficied, program will automatically store them as 1.png and 2.png in the cuurent directory")
        f = h5py.File('data/test.h5', 'r')
        test = f['points'][:]
        test_1000 = graph_construct.pts_sample(test, 1000)
        test_1000 = graph_construct.pts_norm(test_1000)
        A = graph_construct.graph_construct_kneigh(test_1000)[0]
        G = visual.graph(A)
        visual.draw_graph(G, test_1000, path='1.png')
        visual.visual(test_1000, path='2.png')
        f.close()
    elif args.testorall == 'all':
        print('If you are first time running this project, please run "python run.py --mode download"')
        print('You can specify which graph construction method to use by using --method and follow the corresponding param.')
        print('For example, python run.py --mode download --method knn --k 15, if no method specified, default is knn with k=30')
        print('Raw modelnet data is stored in data/modelnet/ModelNet40, points data will be stored in data/modelnet/modelnet_points')
        print('Processed data will stored in data/modelnet/ with corresponding method name')
        print('ALERT: Do not move the data files, it may cause problems')
        if args.mode == 'download':
            #print('Download the data')
            #create.download()
            print('Create the points')
            create.create_pt()
            print('Construct Graph')
            if args.method == 'knn':
                create.construct_graph_knn(k=args.k)
            elif args.method =='fix_radius':
                create.construct_graph_radius(r=args.r)
            else:
                print('only support knn and fix_radius')
            print('Done creating dataset! Now see the readme for instruction for model training')
        if args.mode == 'train':
            temp = {
                'data': args.data,
                'base': args.base,
                'model': args.model,
                'pool': args.pool,
                'ratio': args.ratio,
                'val_size': args.val_size,
                'lr': args.lr,
                'epoch': args.epoch,
                'bs': args.bs,
                'model_path': args.model_path
            }
            temp = run(temp)
            temp_out = args.output_path
            print(temp)
            print('Saving results in ' + temp_out)
            test = temp.process()
            out = pd.DataFrame()
            out['epoch'] = [x[0] for x in test]
            out['train_ls'] = [x[1] for x in test]
            out['test_ls'] = [x[2] for x in test]
            out['test_acc'] = [x[3] for x in test]
            out.to_csv(temp_out, index=False)
    else:
        print('only support test and all')
if __name__ == '__main__':
    main()