#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Basic Matrix Factorization using just numpy in single run
    @author <ariel kalingking> akalingking@gmail.com """
import sys
import numpy as np
from scipy import sparse
import pandas as pd
import metrics
import metrics2
if sys.version_info[0] < 3:
    range = xrange

def main():
    print("\nStarting '%s'" % sys.argv[0])

    np.random.seed(8000)

    """ Load dataset """
    datafile = "./data/ml-100k/u.data"
    data = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

    """ Convert rating data to n_user x n_item matrix format """
    data = data.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)

    users = np.unique(ratings.index.values)
    items = np.unique(ratings.columns.values)
    n_users = len(users)
    n_items = len(items)
    print ("n_users=%d n_items=%d" % (n_users, n_items))

    """ Take the mean only from non-zero elements """
    temp = ratings.copy()
    rating_mean = temp.copy().replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print("Rating mean:%.3f" % rating_mean)

    """ Find PQ sub matrices """
    R = ratings.values

    """ Randomly initialize P & Q matrices with n latent factors """
    n_factors = 10
    P = np.random.normal(0, .1, (n_users, n_factors))
    Q = np.random.normal(0, .1, (n_factors, n_items))

    R_mask = R.copy()
    R_mask[R_mask != 0.000000] = 1
    R_hat = np.zeros(np.shape(R))
    R_hat_mask = np.zeros(np.shape(R))
    np.matmul(P, Q, out=R_hat)

    # Get errors only from explicitly rated elements
    np.multiply(R_hat, R_mask, out=R_hat_mask)

    """ Compute error: MSE = (1/N) * (R - R_hat), RMSE = MSE^(1/2) """
    diff = np.subtract(R, R_hat_mask)
    diff_square = np.square(diff)
    mse = np.divide(diff_square.sum(), n_users*n_items)
    rmse = np.sqrt(mse)

    print ("RMSE: %.5f" % (rmse))

    ratings_csr = sparse.csr_matrix(ratings.values)
    precision = metrics2.precision_at_k(ratings_csr, R_hat, k=100)
    recall = metrics2.recall_at_k(ratings_csr, R_hat, k=100)
    print ("Precision {0:.3f}% Recall={1:.3f}%".format(precision*100, recall*100))

    print("\nStopping '%s'" % sys.argv[0])


if __name__ == "__main__":
    main()
