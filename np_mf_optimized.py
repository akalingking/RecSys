#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief Matrix factorization with opptimization using gradient descent
    @author: <ariel kalingking> akalingking@gmail.com """
import time
import numpy as np
import pandas as pd
from metrics import precision_at_k, recall_at_k

def main():
    np.random.seed(8000)

    """ Load dataset """
    datafile = "./data/ml-100k/u.data"
    data = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

    """ Convert rating data to user x movie matrix format"""
    data = data.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)

    train_size = 0.5
    train_row_size = int(len(ratings.index) * train_size)
    train_col_size = int(len(ratings.columns) * train_size)
    ratings = ratings.loc[:train_row_size, :train_col_size]
    user_train = np.unique(ratings.index.values)
    item_train = np.unique(ratings.columns.values)
    n_users = len(user_train)
    n_items = len(item_train)
    print ("n_users:%d" % n_users)
    print ("n_items:%d" % n_items)

    """ Take the mean only from non-zero elements """
    temp = ratings.copy()
    rating_mean = temp.replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print("Rating mean: %.2f" % rating_mean)

    """ Optimization parameters """
    epoch = 50
    n_factors = 10
    learning_rate = .001
    lambda_ = .0001

    """ Find PQ sub matrices """
    R = ratings.values
    P = np.random.normal(0, .1, (n_users, n_factors))
    Q = np.random.normal(0, .1, (n_factors, n_items))

    print("Start optimization...")
    verbosity = 1
    R_mask = R.copy()
    R_mask[R_mask != 0.000000] = 1
    R_hat = np.zeros(np.shape(R))
    R_hat_mask = np.zeros(np.shape(R))
    start_t = time.time()
    for iteration in xrange(epoch):
        for u in xrange(n_users):
            for i in xrange(n_items):
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
        if iteration%verbosity == 0 or iteration == (epoch-1):
            print ("Epoch %d: RMSE: %.5f" % (iteration, rmse))

    print ("Optimization time: %.6f seconds" % (time.time()- start_t))
    
    """ Metrics of recommended items for each user """
    k = 100
    precisions, recalls = [], []
    for i in xrange(n_users):
        R_u = ratings.values[i, :]
        R_u_k = R_u[:k]
        R_u_k_non_zero = R_u_k[R_u_k > 0]
        if len(R_u_k_non_zero) > 0:
            # print ("User:%d mean rating:%.5f" % (i+1, np.mean(R_u_k_non_zero)))
            R_u_relevant = np.where(R_u_k > rating_mean)[0]
            R_u_hat = R_hat[i, :]
            R_u_hat_sorted = np.argsort(-R_u_hat)
            assert (R_u.shape == R_u_hat.shape)
            precision = precision_at_k(R_u_relevant, R_u_hat_sorted[:k], k=k)
            precisions.append(precision)
            recall = recall_at_k(R_u_relevant, R_u_hat_sorted[:k], k=k)
            recalls.append(recall)
            #print ("user:%d precision:%.6f recall:%.6f" % (i+1, precision, recall))

    print ("Precision:%.3f Recall:%.6f" % (np.mean(precisions), np.mean(recalls)))

if __name__=="__main__":
    main()