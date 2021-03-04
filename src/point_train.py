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
from tqdm.notebook import tqdm
class Trainer():
    
    def __init__(self,model,train_set,test_set,opts):
        self.model = model  # neural net
        
        # device agnostic code snippet
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.epochs = opts['epochs']
        print(model)
        self.optimizer = torch.optim.Adam(model.parameters(), opts['lr']) # optimizer method for gradient descent
        self.criterion = torch.nn.CrossEntropyLoss()                      # loss function
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=opts['batch_size'],
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=opts['batch_size'],
                                                       shuffle=True)
        self.stats = []
        self.best_acc = 0
        self.model_path = opts['model_path']
        
    def train(self):
        self.model.train() #put model in training mode
        for epoch in range(self.epochs):
            self.tr_loss = []
            for i, (x, labels) in tqdm(enumerate(self.train_loader),
                                                   total = len(self.train_loader)):
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                #print(labels)
                outputs = self.model(x)  
                loss = self.criterion(outputs, labels) 
                loss.backward()                        
                self.optimizer.step()                  
                self.tr_loss.append(loss.item())       
            
            self.test(epoch) # run through the validation set
        
    def test(self,epoch):
            
            self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
            self.test_loss = []
            self.test_accuracy = []
            
            for i, (x, labels) in enumerate(self.test_loader):
                
                x, labels = x.to(self.device), labels.to(self.device)
                # pass data through network
                # turn off gradient calculation to speed up calcs and reduce memory
                with torch.no_grad():
                    outputs = self.model(x)
                
                # make our predictions and update our loss info
                _, predicted = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                self.test_loss.append(loss.item())
                
                self.test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            
            print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format( 
                  epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss), np.mean(self.test_accuracy)))
            self.stats.append((epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss), np.mean(self.test_accuracy)))
            temp_acc = np.mean(self.test_accuracy)
            if temp_acc > self.best_acc:
                print('Found better. Saving model dict')
                self.best_acc = temp_acc
                torch.save(self.model.state_dict(), self.model_path)

    def get_stats(self):
        return self.stats