#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Matrix factorization with optimization using just numpy and gradient descent
    @author <ariel kalingking> akalingking@gmail.com """
import time, sys
import numpy as np
import pandas as pd
import metrics
import utils

from scipy import sparse

def main():
    print("\nStarting '%s'" % sys.argv[0])

    np.random.seed(8000)

    """ Load dataset """
    datafile = "./data/ml-100k/u.data"
    data = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

    """ Convert rating data to user x movie matrix format"""
    data = data.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)

    users = np.unique(ratings.index.values)
    items = np.unique(ratings.columns.values)
    n_users = len(users)
    n_items = len(items)
    print ("n_users:%d n_items=%d" % (n_users, n_items))

    """ Take the mean only from non-zero elements """
    temp = ratings.copy()
    rating_mean = temp.replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print("Rating mean: %.3f" % rating_mean)

    """ Optimization parameters """
    epoch = 10
    n_factors = 10
    learning_rate = .001
    lambda_ = .0001

    """ Find PQ sub matrices """
    R = ratings.values
    P = np.random.normal(0, .1, (n_users, n_factors))
    Q = np.random.normal(0, .1, (n_factors, n_items))

    print ("Start gradient descent...")
    verbosity = 1
    R_mask = R.copy()
    R_mask[R_mask != 0.000000] = 1
    R_hat = np.zeros(np.shape(R))
    R_hat_mask = np.zeros(np.shape(R))
    start_t = time.time()
    for iteration in range(epoch):
        for u in range(n_users):
            for i in range(n_items):
                error = R[u, i] - np.matmul(P[u], Q[:,i])
                # Update process:
                # P_i = P_i - learning_rate * error * derivative{ loss_fn }
                # where: loss_fn = (R - PQ)ˆ2 + lambda (|P|^2 + |Q|ˆ2)
                P[u] = P[u] - learning_rate * ((error * -2 * Q[:, i]) + (lambda_ * 2 * np.abs(Q[:,i])))
                Q[:,i] = Q[:, i] - learning_rate * ((error * -2 * P[u]) + (lambda_ * 2 * np.abs(P[u])))

        np.matmul(P, Q, out=R_hat)

        # Get errors only from explicitly rated elements
        np.multiply(R_hat, R_mask, out=R_hat_mask)

        # Compute error: MSE = (1/N) * (R - Rˆ), RMSE = MSEˆ(1/2)
        diff = np.subtract(R, R_hat_mask)
        diff_square = np.square(diff)
        mse = np.divide(diff_square.sum(), n_users*n_items)
        rmse = np.sqrt(mse)
        if iteration % verbosity == 0 or iteration == (epoch-1):
            print ("Epoch %d: RMSE: %.5f" % (iteration, rmse))

    print ("Optimization time: %.6f seconds" % (time.time()- start_t))
    
    """ Metrics of recommended items for each user """
    assert (ratings.shape == R_hat.shape)
    k = 100
    interactions = sparse.csr_matrix(ratings.values)
    predicted_ranks = utils.rank_matrix(R_hat)

    precision = metrics.precision_at_k(predicted_ranks, interactions, k=k)
    recall = metrics.recall_at_k(predicted_ranks, interactions, k=100)

    print("Precision:%.3f%% Recall:%.3f%%" % (precision*100, recall*100))

    print("\nStopping '%s'" % sys.argv[0])


if __name__=="__main__":
    main()