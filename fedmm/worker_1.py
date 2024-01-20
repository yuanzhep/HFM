#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fedmm, yz
01/09/2024, silo
"""

import numpy as np
from fedmm.client_CD import LinearCDClient, LogisticCDClient

def getClientClass(clientType):
    if clientType == "linear":
        return LinearCDClient
    elif clientType == "logistic":
        return LogisticCDClient
    else:
        raise Exception()

class Hub(object):
    def __init__(self, theta, alpha: float, Xtheta: np.array, X: np.array, y : np.array,
                 index: int, offset: int, device_list: list, clientType: str) -> None:
        self.alpha: float = alpha
        self.Xtheta: np.array = Xtheta
        self.costs = []
        self.theta = theta
        self.X = X
        self.y = y
        self.index = index
        self.theta_average = self.theta[index*offset : (index+1)*offset]
        self.local_estimate = None
        self.device_list = device_list
        self.global_Xtheta=None
        self.client_class = getClientClass(clientType)