#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Precision and Recall from
            https://en.wikipedia.org/wiki/Precision_and_recall 
    @author <ariel kalingking> akalingking@gmail.com
    """
import numpy as np

def recall_at_k (data, predicted, k=0, debug_on=False):
    assert (isinstance(data, np.ndarray))
    assert (isinstance(predicted, np.ndarray))
    if debug_on:
        print ("recall_at_k data:%d predicted:%d k:%d" % (len(data), len(predicted), k))

    k = k if k>0 else len(predicted)

    positive = data[:k]
    predicted_ = predicted[:k]

    """ True Positive: correctly classified """
    tp = np.intersect1d(positive, predicted_)
    TP = len(tp)

    """ False Negative: classified as negative but is positive """
    FN = len(data) - TP

    recall = (TP * 1.0) / (TP + (FN if FN > 0 else 0)) if TP > 0 else 0

    if debug_on:
        print("recall_at_k TP:%d FN:%d recall:%.5f" % (TP, FN, recall))

    return recall

def precision_at_k (data, predicted, k=0, debug_on=False):
    assert (isinstance(data, np.ndarray))
    assert (isinstance(predicted, np.ndarray))

    if debug_on:
        print ("precision_at_k data:%d predicted:%d k:%d" % (len(data), len(predicted), k))

    k = k if k > 0 else len(predicted)

    positive = data[:k]
    predicted_ = predicted[:k]

    """ True Positive: correctly classified """
    tp = np.intersect1d(positive, predicted_)
    TP = len(tp)

    """ False Positive: classified as positive but is negative """
    FP = len(predicted_) - TP

    precision = (TP * 1.0) / (TP + FP)
    if debug_on:
        print("precision_at_k TP:%d FP:%d precision:%.5f " % (TP, FP, precision))

    return precision