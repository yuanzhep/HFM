#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fedmm, yz
01/09/2024, worker
"""

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(618)

class Worker(object):
    def __init__(self, n_epochs: int, batch_size: int, learning_rate: float, X: np.array, y: np.array,
                 offset: float, model: nn.Module, client_index: int, dc_index: int):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.offset = offset
        self.model = model
        self.client_index = client_index
        self.dc_index = dc_index
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.n_batches = int(np.ceil(len(X) / self.batch_size))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, n_epochs=0, device=""):
        if n_epochs > 0:
            self.n_epochs = n_epochs
        self.model.train()
        self.model.to(device)
        for ep in range(self.n_epochs):
            for batch_X, batch_Y in self.dataloader:
                self.optimizer.zero_grad()
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                pred_Y = self.model(batch_X)
                loss = self.criterion(pred_Y, batch_Y)
                loss.backward()
                self.optimizer.step()

        return self.model

    def set_model(self, global_model):
        self.model = copy.deepcopy(global_model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)