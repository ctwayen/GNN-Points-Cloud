import torch
import os
from src import graph_trainer
from src import graph_dataset
from src import graph_model
import json
import pandas as pd
import os
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
            model = graph_model.GCN(pool=args['pool'], ratio=args['ratio'], class_num=class_num)
            opts = {
                    'lr': args['lr'],
                    'epochs': args['epoch'],
                    'batch_size': args['bs']
            }
        self.Train = graph_trainer.trainer(model = model,
                      train_set = train_dataset,
                      test_set = test_dataset,opts = opts)
    def process(self):
        self.Train.train()
        return self.Train.get_stats()
    
def main():
    params_base = 'config/model_params/GCN_k_15/'
    output_base = 'config/model_results/GCN_k_15/'
    for obj in os.listdir(params_base):
        temp_out = output_base + obj[:-5] + '.csv'
        temp_in = params_base + obj
        print(temp_out, temp_in)
        with open(temp_in, 'r') as fp:
            temp_dct = json.load(fp)
        temp = run(temp_dct)
        test = temp.process()
        out = pd.DataFrame()
        out['epoch'] = test[0]
        out['train_ls'] = test[1]
        out['test_ls'] = test[2]
        out['test_acc'] = test[3]
        out.to_csv(temp_out, index=False)

if __name__ == '__main__':
    main()