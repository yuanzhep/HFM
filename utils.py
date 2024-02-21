#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yz
01/13/2024, fedmm silo avg
01/24/2024, fix m3 index issue 
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
import random
import copy
import os
from tqdm import tqdm
from PIL import Image
from print_metrics import print_metrics_binary
from sklearn.utils import shuffle

def average_models(model, silo_models):
    num_silo = len(silo_models)
    theta_sum = np.zeros(model.shape)
    for theta in silo_models:
        theta_sum += theta
    model = theta_sum / num_silo
    return model

def normalize(x, means=None, stds=None):
    num_dims = x.shape[1]
    if means is None and stds is None:
        means = []
        stds = []
        for dim in range(num_dims):
            m = x[:, dim, :, :].mean()
            st = x[:, dim, :, :].std()
            x[:, dim, :, :] = (x[:, dim, :, :] - m)/st
            means.append(m.item())
            stds.append(st.item())
        return x , means, stds
    else:
        for dim in range(num_dims):
            m = means[dim]
            st = stds[dim]
            x[:, dim, :, :] = (x[:, dim, :, :] - m)/st
        return x , None, None
    
class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y, index

    def __len__(self):
        return self.tensors[0].size(0)
    
class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None, perform_transform=False, datapoints=0, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.x = []
        self.y = []
        self.root = root
        self.x, self.y = shuffle(self.x,self.y, random_state=seed)

        self.classes, self.class_to_idx = self.find_classes(root)
        self.transform = transform
        self.target_transform = target_transform
        self.perform_transform = perform_transform
        self.datapoints = datapoints
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label + '/' + data_type):
                views = []
                for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                    views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)

                self.x.append(views)
                self.y.append(self.class_to_idx[label])
                
        if datapoints>0:
            self.x = self.x[:self.datapoints]
            self.y = self.y[:self.datapoints]
        
        if perform_transform:
            self.x = self.transformDataset(self.x, self.transform)
        
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []
        if not self.perform_transform:
            for view in orginal_views:
                im = Image.open(view)
                im = im.convert('RGB')
                if self.transform is not None:
                    im = self.transform(im)
                views.append(im)
    
            return views, self.y[index], index
        else:
            return orginal_views, self.y[index], index
                     
    def __len__(self):
        return len(self.x)
    
    def transformDataset(self, data, transform):
        print("Transforming Dataset using ", transform)
        res = []
        for sample in tqdm(data):
            images = []
            for view in sample:
                im = Image.open(view)
                im = im.convert('RGB')
                im = transform(im)
                images.append(im)
            res.append(images)
        return res
  
class TopLayer(nn.Module):
    def __init__(self, linear_size=512, nb_classes=10, bias = False):
        super(TopLayer, self).__init__()
        self.classifier = nn.Linear(256+256, nb_classes, bias=bias)
        
    def forward(self, x):
        x = self.classifier(F.relu(x))
        return x
    
class MIMICIII_LSTM_combined(nn.Module):
    def __init__(self, dim, input_dim, dropout=0.0, num_classes=1,
                 num_layers=1, batch_first=True, **kwargs):
        super(MIMICIII_LSTM_combined, self).__init__()
        self.hidden_dim = dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=self.hidden_dim,
                            batch_first=batch_first)
        
        # self.initialize_weights(self.lstm)
        self.do = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_dim, num_classes) 
        
    def forward(self, x):
        training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm.flatten_parameters()
        lstm_out2, (h2, c2) = self.lstm(self.do(x))
        h2 = h2.view(-1, self.hidden_dim) 
        output = self.linear(h2)
        return output
         
    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

class MIMICIII_LSTM(nn.Module):
    def __init__(self, dim, input_dim, dropout=0.0, num_classes=1,
                 num_layers=1, batch_first=True, **kwargs):
        super(MIMICIII_LSTM, self).__init__()
        self.hidden_dim = dim
        self.input_dim = input_dim
        self.num_layers = num_layers
    
        self.biLSTM = nn.LSTM(input_size=self.input_dim, 
                              hidden_size=self.hidden_dim//2, 
                              num_layers=1, 
                              bidirectional=True, 
                              batch_first=batch_first)
        
        self.lstm = nn.LSTM(input_size=self.hidden_dim, 
                            hidden_size=self.hidden_dim,
                            batch_first=batch_first)
        
        self.initialize_weights(self.biLSTM)
        self.initialize_weights(self.lstm)
        
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.biLSTM.flatten_parameters()
        h0 = torch.zeros(2, x.size(0), self.hidden_dim//2).to(training_device) 
        c0 = torch.zeros(2, x.size(0), self.hidden_dim//2).to(training_device)
        lstm_out1, (h1, c1) = self.biLSTM(x, (h0, c0))
        self.lstm.flatten_parameters()
        lstm_out2, (h2, c2) = self.lstm(self.do(lstm_out1))
        h2 = h2.view(-1, self.hidden_dim) 
        output = h2
        return output
        
    def initialize_weights(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

class MIMICIII_lstm_toplayer(nn.Module):
    def __init__(self, linear_size=64, nb_classes=1, dropout= 0.0, bias = False):
        super(MIMICIII_lstm_toplayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(linear_size, nb_classes, bias=bias)
        
    def forward(self, x):
        x = self.classifier(self.dropout(x))
        return x
    
def add_model(dst_model, src_model):
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data + dict_params2[name1].data)
    return dst_model

def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model

def federated_avg(models: Dict[Any, torch.nn.Module]) -> torch.nn.Module:
    nr_models = len(models)
    model_list = list(models.values())
    device = torch.device('cuda' if next(model_list[0].parameters()).is_cuda else 'cpu')

    model = copy.deepcopy(model_list[0])
    model.to(device)
    model = scale_model(model, 0.0)

    for i in range(nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model

def get_train_or_test_loss(network_left,network_right,overall_top_layer,
                           overall_train_dataloader, 
                           overall_test_dataloader, report, cord_div_idx=16):
    network_left.eval()
    network_right.eval()
    overall_top_layer.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            input_top_layer = torch.cat((output_left, output_right), dim=1)
            output_top = overall_top_layer(input_top_layer)

            train_loss += F.cross_entropy(output_top, target.long()).item()
            pred = output_top.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.long().data.view_as(pred)).sum()
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))

    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            input_top_layer = torch.cat((output_left, output_right), dim=1)
            output_top = overall_top_layer(input_top_layer)

            test_loss += F.cross_entropy(output_top, target.long()).item()
            pred = output_top.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.long().data.view_as(pred)).sum()
        test_loss /= len(overall_test_dataloader)
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))

def general_get_train_or_test_loss_lstm(networks:List,
                                   overall_top_layer,
                                   overall_train_dataloader, 
                                   overall_test_dataloader, 
                                   report, coordinate_partitions=None):
    num_parties = len(networks)
    for network in networks:
        network.eval()
    overall_top_layer.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    targets = []
    probabilities = []

    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:           
            vert_data = [data[:, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)

            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = overall_top_layer(input_top_layer)[:, 0]

            train_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            pred = output_top.squeeze()>0.0 
            train_correct += pred.float().eq(target.float()).sum() 
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["train_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))
             
    targets = []
    probabilities = []
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            vert_data = [data[:, :,coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)
            
            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = overall_top_layer(input_top_layer)[:, 0]

            test_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            pred = output_top.squeeze()>0.0
            test_correct += pred.float().eq(target.float()).sum() 
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["test_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))

def general_get_train_or_test_loss_lstm_combined(networks:List,
                                   overall_train_dataloader, 
                                   overall_test_dataloader, 
                                   report, coordinate_partitions=None):
    num_parties = len(networks)
    for network in networks:
        network.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    targets = []
    probabilities = []
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:           
            vert_data = [data[:, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)

            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = input_top_layer.sum(dim=1)

            train_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            pred = output_top.squeeze()>0.0
            train_correct += pred.float().eq(target.float()).sum() 
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["train_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))
               
    targets = []
    probabilities = []
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            vert_data = [data[:, :,coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)
            
            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]

            input_top_layer = torch.cat(H_embeddings, dim=1)
            output_top = input_top_layer.sum(dim=1)
            
            test_loss += F.binary_cross_entropy_with_logits(output_top, target.float()).item()
            pred = output_top.squeeze()>0.0
            test_correct += pred.float().eq(target.float()).sum() 
            
            targets.append(target.float().cpu())
            probabilities.append(torch.sigmoid(output_top.detach().cpu()))
            
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        probabilities = torch.cat(probabilities)    
        targets = torch.cat(targets)
        report["test_ret"].append(print_metrics_binary(targets, probabilities))
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))
         
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0))
        return res
               
def get_train_or_test_loss_generic(networks:List,
                                   overall_train_dataloader, 
                                   overall_test_dataloader, 
                                   report, coordinate_partitions=None):
    num_parties = len(networks)
    for network in networks:
        network.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    train_correct5 = 0
    test_correct5 = 0
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:
            vert_data = [data[:, :, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)
            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]
            input_top_layer = torch.stack(H_embeddings)
            output_top = input_top_layer.sum(dim=0)

            train_loss += F.cross_entropy(output_top, target.long()).item()
            pred = output_top.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.long().data.view_as(pred)).sum()
            train_correct5 += accuracy(output_top, target.long(), topk=(5,))[0]
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        report["train_accuracy5"].append(train_correct5 / len(overall_train_dataloader.dataset))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset),
            train_correct5 / len(overall_train_dataloader.dataset)))

    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            vert_data = [data[:, :, :, coordinate_partitions[i]] for i in range(num_parties)]
            for i in range(len(vert_data)):
                vert_data[i] = vert_data[i].to(device)
            target = target.to(device)

            H_embeddings = [networks[i](vert_data[i]) for i in range(num_parties)]
            
            input_top_layer = torch.stack(H_embeddings)
            output_top = input_top_layer.sum(dim=0)
            
            test_loss += F.cross_entropy(output_top, target.long()).item()
            pred = output_top.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.long().data.view_as(pred)).sum()
            test_correct5 += accuracy(output_top, target.long(), topk=(5,))[0]
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        report["test_accuracy5"].append(test_correct5 / len(overall_test_dataloader.dataset))       
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset),
            test_correct5 / len(overall_test_dataloader.dataset)))        
          
def get_train_or_test_loss_simplified_cifar(network_left,network_right,overall_train_dataloader, 
                           overall_test_dataloader, report, cord_div_idx=16):
    network_left.eval()
    network_right.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            output_top = output_right + output_left

            train_loss += F.cross_entropy(output_top, target.long()).item()
            pred = output_top.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.long().data.view_as(pred)).sum()
        train_loss /= len(overall_train_dataloader)
        
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))

    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            output_top = output_right + output_left

            test_loss += F.cross_entropy(output_top, target.long()).item()
            pred = output_top.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.long().data.view_as(pred)).sum()
        test_loss /= len(overall_test_dataloader)
        
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))

# if __name__ == "__main__":
#     one = nn.Conv2d(20, 13, 3)
#     two =nn.Conv2d(20, 13, 3)
#     three = nn.Conv2d(20, 13, 3)
#     bb = federated_avg({1:one, 2:two, 3:three})
#     assert torch.isclose(bb.weight.data, (one.weight.data + two.weight.data + three.weight.data)/3.0).sum() == bb.weight.data.numel()
