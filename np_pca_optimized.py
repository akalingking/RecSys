#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Recommendation algorithm using Principal Component Analysis (PCA)
    @author <ariel kalingking> akalingking@gmail.com """
import sys
import numpy as np
import pandas as pd
import metrics2
from scipy import sparse

if sys.version_info[0] < 3:
    range = xrange

def main():
    print ("\nStarting '%s'" % sys.argv[0])

    np.random.seed(8000)
    normalization_enabled = False
    optimize_enabled = True
    k = 100

    """ Load dataset """
    datafile = "./data/ml-100k/u.data"
    data = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

    """ Convert rating data to user x movie matrix format """
    data = data.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)

    """ Construct data """
    users = np.unique(ratings.index.values)
    items = np.unique(ratings.columns.values)
    n_users = len(users)
    n_items = len(items)
    print ("n_users=%d n_items=%d" % (n_users, n_items))

    """ Compute mean ratingonly from non-zero elements """
    temp = ratings.copy()
    rating_mean = temp.copy().replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print("Rating mean: %.6f" % rating_mean)

    R_mask = np.zeros(np.shape(ratings))
    R_mask[ratings != 0.000000] = 1

    if normalization_enabled:
        temp = ratings.copy()
        ratings_norm = np.subtract(temp, rating_mean, where=temp!=0)
        ratings_norm = np.multiply(ratings_norm, R_mask)
        assert (np.count_nonzero(ratings_norm) == np.count_nonzero(ratings))
        R = ratings_norm.values
    else:
        R = ratings.values.copy()

    # Setup covariance to treat the item columns as input variables
    covar = np.cov(R, rowvar=False)
    evals, evecs = np.linalg.eigh(covar)

    print ("cov_mat shape: %s" % str(np.shape(covar)))
    print ("evals shape: %s" % str(np.shape(evals)))
    print ("evecs shape: %s" % str(np.shape(evecs)))

    n_components = 10 # principal components

    """ Randomly initialize weights table """
    weights = np.random.normal(0, .1, (n_users, n_components))
    components = evecs[:n_components, :n_items]

    R_hat_mask = np.zeros(np.shape(R), dtype=np.float64)

    if optimize_enabled:
        # optimization parameters
        epochs = 5
        learning_rate = .0001
        lambda_ = .0001
        verbosity = 1
        print ("Optimized PCA epochs=%s" % epochs)
        """ We only modify the weight matrix """
        for epoch in range(epochs):
            for u in range(n_users):
                for i in range(n_items):
                    error = R[u, i] - np.dot(weights[u, :], components[:, i])
                    for k in range(n_components):
                        weights[u, k] = weights[u, k] - learning_rate * (error * -2 * components[k, i] + lambda_ * (2*np.abs(weights[u, k]) + 2*np.abs(components[k,i])))

            R_hat = np.zeros(np.shape(R))
            np.matmul(weights, components, out=R_hat)
            # Get errors only from explicitly rated elements
            np.multiply(R_hat, R_mask, out=R_hat_mask)
            # Compute error: MSE = (1/N) * (R - Rˆ), RMSE = MSEˆ(1/2)
            diff = np.subtract(R, R_hat_mask)
            diff_square = np.square(diff)
            mse = np.divide(diff_square.sum(), np.count_nonzero(R))
            rmse = np.sqrt(mse)
            if epoch % verbosity == 0 or epoch == (epochs - 1):
                print ("Epoch %d: RMSE: %.6f" % (epoch, rmse))
    else:
        R_hat = np.matmul(weights, components)
        print("R_hat shape: %s" % str(np.shape(R_hat)))
        assert (np.shape(R) == np.shape(R_hat))

        print ("PCA single run")
        np.multiply(R_hat, R_mask, out=R_hat_mask)
        # Compute error: MSE = (1/N) * (R - Rˆ), RMSE = MSEˆ(1/2)
        diff = np.subtract(R, R_hat_mask)
        diff_square = np.square(diff)
        mse = np.divide(diff_square.sum(), np.count_nonzero(R))
        rmse = np.sqrt(mse)
        print ("RMSE: %.5f" % rmse)

    sparse_data = sparse.csr_matrix(R)
    precision = metrics2.precision_at_k(sparse_data, R_hat, k=k)
    recall = metrics2.recall_at_k(sparse_data, R_hat, k=k)
    print("Precision:%.3f%% Recall:%.3f%%" % (precision*100, recall*100))

    print ("\nStoppping '%s" % sys.argv[0])


if __name__ == "__main__":
    main()
