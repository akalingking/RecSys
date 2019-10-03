#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Recommendation algorithm using Singular Value Decomposition (SVD)
    @author <ariel kalingking> akalingking@gmail.com """
import sys
import numpy as np
import pandas as pd
import metrics
import scipy.sparse.linalg as linalg
from scipy import sparse

def main():
    print("\nStarting '%s'" % sys.argv[0])

    np.random.seed(8000)

    k = 100

    normalization_enabled = False

    """ Load dataset """
    datafile = "./data/ml-100k/u.data"
    data = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

    """ Convert rating data to user x movie matrix format """
    data = data.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)

    # train_size = 0.7
    # train_row_size = int(len(ratings.index) * train_size)
    # train_col_size = int(len(ratings.columns) * train_size)
    # ratings = ratings.loc[:train_row_size, :train_col_size]
    users = np.unique(ratings.index.values)
    items = np.unique(ratings.columns.values)
    n_users = len(users)
    n_items = len(items)
    assert (np.max(users) == len(users))
    assert (np.max(items) == len(items))
    print ("n_users=%d n_items=%d" % (n_users, n_items))

    """ Take the mean only from non-zero elements """
    temp = ratings.copy()
    rating_mean = temp.copy().replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print("Rating mean: %.2f" % rating_mean)

    if normalization_enabled:
        temp = ratings.copy()
        ratings_norm = np.subtract(temp, rating_mean, where=temp!=0)
        R = ratings_norm.values
    else:
        R = ratings.values

    U, S, V = linalg.svds(R, k=k)
    # print ("U: ", np.shape(U))
    # print ("S: ", np.shape(S))
    # print ("V: ", np.shape(V))
    sigma = np.diag(S)
    # print ("Sigma: ", np.shape(sigma))

    """ Generate prediction matrix """
    R_hat = np.dot(np.dot(U, sigma), V)
    assert (np.shape(R) == np.shape(R_hat))

    # Get errors only from explicitly rated elements
    R_mask = np.zeros(np.shape(R))
    R_mask[R != 0.000000] = 1
    R_hat_mask = np.zeros(np.shape(R))
    np.multiply(R_hat, R_mask, out=R_hat_mask)

    # Compute error: MSE = (1/N) * (R - Rˆ), RMSE = MSEˆ(1/2)
    assert (np.count_nonzero(R) == np.count_nonzero(R_hat_mask))
    diff = np.subtract(R, R_hat_mask)
    diff_square = np.square(diff)
    #mse = np.divide(diff_square.sum(), n_users*n_items)
    mse = np.divide(diff_square.sum(), np.count_nonzero(R_mask))
    rmse = np.sqrt(mse)
    print ("RMSE: %.6f" % (rmse))

    assert (R.shape == R_hat.shape)
    interactions = sparse.csr_matrix(R)
    predicted_ranks = metrics.rank_matrix(R_hat)
    precision = metrics.precision_at_k(predicted_ranks, interactions, k=k)
    recall = metrics.recall_at_k(predicted_ranks, interactions, k=k)
    print("Precision:%.3f%% Recall:%.3f%%" % (precision * 100, recall * 100))

    print("\nStopping '%s'" % sys.argv[0])


if __name__ == "__main__":
    main()
