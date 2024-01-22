#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fedmm, yz
01/14/2024, acc metrics
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics

def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    m40 = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(m40)
    m40 = m40.astype(np.float32)

    acc = (m40[0][0] + m40[1][1]) / np.sum(m40)
    prec0 = m40[0][0] / (m40[0][0] + m40[1][0])
    prec1 = m40[1][1] / (m40[1][1] + m40[0][1])
    rec0 = m40[0][0] / (m40[0][0] + m40[0][1])
    rec1 = m40[1][1] / (m40[1][1] + m40[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    f1_score = metrics.f1_score(y_true, predictions.argmax(axis=1))
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("F1 Score = {}".format(f1_score))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse,
            "f1_score": f1_score}
