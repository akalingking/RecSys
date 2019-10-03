#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Precision and Recall
    @ref    https://en.wikipedia.org/wiki/Precision_and_recall
            Modified from https://github.com/lyst/lightfm and https://github.com/jfkirk/tensorrec
    @author <ariel kalingking> akalingking@gmail.com """
import numpy as np
import pandas as pd
from scipy import sparse

__all__ = ["precision_at_k", "recall_at_k", "rank_matrix"]

def rank_matrix(mat):
    assert isinstance(mat, (np.ndarray,))
    mat_ = pd.DataFrame(data=mat)
    mat_ = mat_.rank(axis=1, ascending=False)
    return mat_.values

def precision_at_k(predicted_ranks, interactions, k=10, preserve_rows=False):
    assert isinstance(interactions, (sparse.coo_matrix, sparse.csr_matrix))
    assert isinstance(predicted_ranks, (np.ndarray,))

    relevant = interactions > 0
    ranks = sparse.csr_matrix(predicted_ranks * relevant.A)
    ranks.data = np.less(ranks.data, (k + 1), ranks.data)

    precision = np.squeeze(np.array(ranks.sum(axis=1))).astype(float) / k

    if not preserve_rows:
        precision = precision[relevant.getnnz(axis=1) > 0]

    return precision.mean()


def recall_at_k(predicted_ranks, interactions, k=10, preserve_rows=False):
    assert isinstance(interactions, (sparse.coo_matrix, sparse.csr_matrix))
    assert isinstance(predicted_ranks, (np.ndarray,))

    relevant = interactions > 0
    ranks = sparse.csr_matrix(predicted_ranks * relevant.A)
    ranks.data = np.less(ranks.data, (k + 1), ranks.data)
    retrieved = np.squeeze(relevant.getnnz(axis=1))
    hit = np.squeeze(np.array(ranks.sum(axis=1)))

    if not preserve_rows:
        hit = hit[relevant.getnnz(axis=1) > 0]
        retrieved = retrieved[relevant.getnnz(axis=1) > 0]

    return (hit.astype(float) / retrieved.astype(float)).mean()
