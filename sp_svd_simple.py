#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief Recommendation algorithm using Singular Value Decomposition (SVD)
    @author: <ariel kalingking> akalingking@gmail.com """
import numpy as np
import pandas as pd
from metrics import precision_at_k, recall_at_k
import scipy.sparse.linalg as linalg

def main():
    np.random.seed(8000)

    k = 100

    normalization_enabled = True

    """ Load dataset """
    datafile = "./data/ml-100k/u.data"
    data = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

    """ Convert rating data to user x movie matrix format """
    data = data.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)

    train_size = 0.7
    train_row_size = int(len(ratings.index) * train_size)
    train_col_size = int(len(ratings.columns) * train_size)
    ratings_train = ratings.loc[:train_row_size, :train_col_size]
    user_train = np.unique(ratings_train.index.values)
    item_train = np.unique(ratings_train.columns.values)
    n_users = len(user_train)
    n_items = len(item_train)
    print ("n_users:", n_users)
    print ("n_items", n_items)

    """ Take the mean only from non-zero elements """
    temp = ratings_train.copy()
    rating_mean = temp.copy().replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print("Rating mean: %.2f" % rating_mean)

    if normalization_enabled:
        temp = ratings_train.copy()
        ratings_norm = np.subtract(temp, rating_mean, where=temp!=0)
        ratings = ratings_norm.values
    else:
        ratings = ratings_train.values

    U, S, V = linalg.svds(ratings, k=5)
    # print ("U: ", np.shape(U))
    # print ("S: ", np.shape(S))
    # print ("V: ", np.shape(V))
    sigma = np.diag(S)
    # print ("Sigma: ", np.shape(sigma))
    R = np.dot(np.dot(U, sigma), V)

    R_mask = R.copy()
    R_mask[R_mask != 0.000000] = 1
    R_hat = np.zeros(np.shape(R))

    # Get errors only from explicitly rated elements
    R_hat_mask = np.zeros(np.shape(R))
    np.multiply(R_hat, R_mask, out=R_hat_mask)
    # Compute error: MSE = (1/N) * (R - Rˆ), RMSE = MSEˆ(1/2)
    diff = np.subtract(R, R_hat_mask)
    diff_square = np.square(diff)
    mse = np.divide(diff_square.sum(), n_users*n_items)
    rmse = np.sqrt(mse)
    print ("RMSE: %.6f" % (rmse))

    """ Metrics for recommended items """
    precisions, recalls = [], []
    for i in xrange(n_users):
        R_u = ratings_train.values[i, :]
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
            #print ("user:%d precision:%.6f recall:%.6f" % (i + 1, precision, recall))

    print ("Precision:%.6f Recall:%.6f" % (np.mean(precisions), np.mean(recalls)))


if __name__ == "__main__":
    main()
