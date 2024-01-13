#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yz
"""
import numpy as np

def average_models(model, client_models):
    num_clients = len(client_models)
    theta_sum = np.zeros(model.shape)
    for theta in client_models:
        theta_sum += theta
    model = theta_sum / num_clients
    
    return model