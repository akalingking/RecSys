#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Recommendation algorithm using Principal Component Analysis (PCA)
    @author <ariel kalingking> akalingking@gmail.com """
import numpy as np
import pandas as pd
from metrics import precision_at_k, recall_at_k

def main():
    np.random.seed(8000)

    normalization_enabled = True

    optimize_enabled = True

    """ Load dataset """
    datafile = "./data/ml-100k/u.data"
    data = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

    """ Convert rating data to user x movie matrix format """
    data = data.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)

    """ Construct training data """
    train_size = 0.7
    train_row_size = int(len(ratings.index) * train_size)
    train_col_size = int(len(ratings.columns) * train_size)
    ratings_train = ratings.loc[:train_row_size, :train_col_size]
    # Obtain unique users and items
    user_train = np.unique(ratings_train.index.values)
    item_train = np.unique(ratings_train.columns.values)
    n_users = len(user_train)
    n_items = len(item_train)
    print ("n_users: %d" % n_users)
    print ("n_items: %d" % n_items)

    """ Compute mean ratingonly from non-zero elements """
    temp = ratings_train.copy()
    rating_mean = temp.copy().replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print("Rating mean: %.6f" % rating_mean)

    if normalization_enabled:
        temp = ratings_train.copy()
        ratings_norm = np.subtract(temp, rating_mean, where=temp!=0)
        ratings = ratings_norm.values
    else:
        ratings = ratings_train.values.copy()

    R = ratings
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

    R_hat = np.matmul(weights, components)
    print ("R_hat shape: %s" % str(np.shape(R_hat)))

    R_mask = R.copy()
    R_mask[R_mask != 0.000000] = 1
    R_hat = np.zeros(np.shape(R), dtype=np.float64)
    R_hat_mask = np.zeros(np.shape(R), dtype=np.float64)

    if optimize_enabled:
        # optimization parameters
        epochs = 50
        learning_rate = .0001
        lambda_ = .0001
        verbosity = 1

        """ We only modify the weight matrix """
        for epoch in xrange(epochs):
            for u in xrange(n_users):
                for i in xrange(n_items):
                    error = R[u, i] - np.dot(weights[u, :], components[:, i])
                    for k in xrange(n_components):
                        weights[u, k] = weights[u, k] - learning_rate * (error * -2 * components[k, i] + lambda_ * (2*np.abs(weights[u, k]) + 2*np.abs(components[k,i])))

            np.matmul(weights, components, out=R_hat)
            # Get errors only from explicitly rated elements
            np.multiply(R_hat, R_mask, out=R_hat_mask)
            # Compute error: MSE = (1/N) * (R - Rˆ), RMSE = MSEˆ(1/2)
            diff = np.subtract(R, R_hat_mask)
            diff_square = np.square(diff)
            mse = np.divide(diff_square.sum(), n_users * n_items)
            rmse = np.sqrt(mse)
            if epoch % verbosity == 0 or epoch == (epochs - 1):
                print ("Epoch %d: RMSE: %.6f" % (epoch, rmse))
    else:
        np.multiply(R_hat, R_mask, out=R_hat_mask)
        # Compute error: MSE = (1/N) * (R - Rˆ), RMSE = MSEˆ(1/2)
        diff = np.subtract(R, R_hat_mask)
        diff_square = np.square(diff)
        mse = np.divide(diff_square.sum(), n_users * n_items)
        rmse = np.sqrt(mse)
        print ("RMSE: %.5f" % rmse)

    """ Metrics for recommended items """
    k = 100
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
            print ("user:%d precision:%.6f recall:%.6f" % (i + 1, precision, recall))

    print ("Precision:%.3f Recall:%.6f" % (np.mean(precisions), np.mean(recalls)))


if __name__ == "__main__":
    main()
