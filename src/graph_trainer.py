import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.utils.data as data
import numpy as np
import h5py
class trainer():
    def __init__(self,model,train_set,test_set,opts):
        self.model = model  # neural net
        
        # device agnostic code snippet
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.epochs = opts['epochs']
        print(model)
        print(opts)
        self.optimizer = torch.optim.Adam(model.parameters(), opts['lr']) # optimizer method for gradient descent
        self.criterion = torch.nn.CrossEntropyLoss()                      # loss function
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=1,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=1,
                                                       shuffle=True)
        self.batch = opts['batch_size']
        self.stats = []
        self.best_acc = 0
        self.model_path = opts['model_path']
        
    def train(self):
        self.model.train() #put model in training mode
        for epoch in range(self.epochs):
            self.tr_loss = []
            for i, (x, edges, weights, labels) in enumerate(self.train_loader):
                x, edges, weights, labels = x.to(self.device), edges.to(self.device), weights.to(self.device), labels.to(self.device)
                #print(labels)
                if i % self.batch == 0 or i==len(self.train_loader):
                    if i != 0:
                        self.optimizer.zero_grad()
                        loss = self.criterion(out, label)
                        loss.backward()                        
                        self.optimizer.step()                  
                        self.tr_loss.append(loss.item())
                    out = self.model(x, edges, weights)
                    label = labels
                else:
                    out = torch.cat((out, self.model(x, edges, weights)), 0)
                    label = torch.cat((label, labels), 0)
            self.test(epoch) # run through the validation set
        
    def test(self,epoch):
            
            self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
            self.test_loss = []
            self.test_accuracy = []
            
            for i, (x, edges, weights, labels) in enumerate(self.test_loader):
                
                x, edges, weights, labels = x.to(self.device), edges.to(self.device), weights.to(self.device), labels.to(self.device)
                # pass data through network
                # turn off gradient calculation to speed up calcs and reduce memory
                with torch.no_grad():
                    outputs = self.model(x, edges, weights)
                
                # make our predictions and update our loss info
                _, predicted = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                self.test_loss.append(loss.item())
                
                self.test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            
            temp_acc = np.mean(self.test_accuracy)
            print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format( 
                  epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss), np.mean(self.test_accuracy)))
            self.stats.append((epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss), temp_acc))
            if temp_acc > self.best_acc:
                print('Found better. Saving model dict')
                self.best_acc = temp_acc
                torch.save(self.model.state_dict(), self.model_path)
            
    def get_stats(self):
        return self.stats