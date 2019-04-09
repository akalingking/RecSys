#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief Recommendation algorithm using optimized Matrix Factorization in TensorFlow
    @author: <ariel kalingking> akalingking@gmail.com """
import numpy as np
import pandas as pd
import tensorflow as tf
from metrics import precision_at_k, recall_at_k

def main():
    session = tf.Session()
    normalized_on = True

    """ load dataset """
    datafile = "./data/ml-100k/u.data"
    df = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])
    n_users = len(np.unique(df.userid))
    n_items = len(np.unique(df.itemid))
    rating_mean = np.mean(df.rating)
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean
    print ("Raw data:")
    print ("Shape: %s" % str(df.shape))
    print ("Userid size: %d" % n_users)
    print ("Itemid size: %d" % n_items)
    print ("Rating mean: %.5f" % rating_mean)

    """ Format ratings to user x item matrix """
    df = df.sort_values(by=["userid", "itemid"])
    ratings = pd.pivot_table(df, values="rating", index="userid", columns="itemid")
    ratings.fillna(0, inplace=True)
    print("Raw ratings size", len(ratings))
    ratings.astype(np.float64)

    """ Construct training data """
    train_size = 0.7
    ratings_train_ = ratings.loc[:int(n_users*train_size), :int(n_items*train_size)]
    user_train = ratings_train_.index.values
    item_train = ratings_train_.columns.values
    n_users = len(user_train)
    n_items = len(item_train)
    temp = ratings_train_.copy()
    rating_mean = temp.replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean

    print ("Training data:")
    print ("Shape: %s" % str(ratings_train_.shape))
    print ("n_users: %d" % n_users)
    print ("n_items: %d" % n_items)
    print ("rating mean: %.5f" % rating_mean)

    user_indices = [x for x in xrange(n_users)]
    item_indices = [x for x in xrange(n_items)]

    print ("Max userid train: ", np.max(user_train))
    print ("Max itemid train", np.max(item_train))
    print ("user_indices size ", len(user_indices))
    print("item_indices size ", len(item_indices))

    if normalized_on:
        ratings_norm = np.zeros(ratings_train_.shape)
        temp = ratings_train_.values
        np.subtract(temp, rating_mean, where=temp!=0, out=ratings_norm)
        ratings = ratings_norm
    else:
        ratings = ratings_train_.values

    # Variables
    n_features = 10 # latent factors
    U = tf.Variable(initial_value=tf.truncated_normal([n_users, n_features]))
    P = tf.Variable(initial_value=tf.truncated_normal([n_features, n_items]))

    result = tf.matmul(U, P)

    result_flatten = tf.reshape(result, [-1])
    assert (result_flatten.shape[0] == n_users * n_items)
    # print ("flatten shape",  result_flatten.shape)

    R = tf.gather(result_flatten, user_indices[:-1] * n_items + item_indices)
    assert (R.shape[0] == n_users*n_items)

    R_ = tf.reshape(R, [tf.div(R.shape[0], n_items), len(item_indices)])
    assert (R_.shape == ratings.shape)

    """ Compute error for values from the original ratings matrix 
        so that means excluding values implicitly computed by UxP """
    var = tf.Variable(ratings.astype(np.float32))
    compare = tf.not_equal(var, tf.constant(0.0))
    compare_op = var.assign(tf.where(compare, tf.ones_like(var), var))
    R_masked = tf.multiply(R_, compare_op)
    assert (ratings.shape == R_masked.shape)

    """ Cost function: sum_ij{ |r_ij- rhat_ij| + lambda*(|u_i|+|p_j|)} """
    diff_op = tf.subtract(ratings.astype(np.float32), R_masked)
    diff_op_abs = tf.abs(diff_op)
    base_cost = tf.reduce_sum(diff_op_abs)

    # Regularizer sum_ij{lambda*(|U_i| + |P_j|)}
    lambda_ = tf.constant(.001)
    norm_sums = tf.add(tf.reduce_sum(tf.abs(U)), tf.reduce_sum(tf.abs(P)))
    regularizer = tf.multiply(norm_sums, lambda_)
    cost = tf.add(base_cost, regularizer)

    """ Optimizer """
    lr = tf.constant(.00001)
    global_step = tf.Variable(0, trainable=False)
    decaying_learning_rate = tf.train.exponential_decay(lr, global_step, 10000, .96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(decaying_learning_rate).minimize(cost, global_step=global_step)

    # """ Run """
    init = tf.global_variables_initializer()
    session.run(init)

    print ("Running stochastic gradient descent..")
    epoch = 200
    for i in range(epoch):
        session.run(optimizer)
        if i%10 == 0 or i == epoch-1:
            diff_op_train = tf.subtract(ratings.astype(np.float32), R_masked)
            diff_op_train_squared = tf.square(diff_op_train)
            se = tf.reduce_sum(diff_op_train_squared)
            mse = tf.divide(se, n_users*n_items)
            rmse = tf.sqrt(mse)
            print("Train iter: %d MSE: %.5f loss: %.5f" % (i, session.run(rmse), session.run(cost)))

    """ Metrics of recommended items for user """
    k = 100
    precisions, recalls = [], []
    # R_hat = session.run(R_[0, :])
    for i in xrange(n_users):
        R_u = ratings_train_.values[i, :]
        R_u_k = R_u[:k]
        R_u_k_non_zero = R_u_k[R_u_k > 0]
        if len(R_u_k_non_zero) > 0:
            # print ("User:%d mean rating:%.5f" % (i+1, np.mean(R_u_k_non_zero)))
            R_u_relevant = np.where(R_u_k > rating_mean)[0]
            R_u_hat = R_[i, :]
            R_u_hat_sorted = np.argsort(-R_u_hat)
            assert (R_u.shape == R_u_hat.shape)
            precision = precision_at_k(R_u_relevant, R_u_hat_sorted[:k], k=k)
            precisions.append(precision)
            recall = recall_at_k(R_u_relevant, R_u_hat_sorted[:k], k=k)
            recalls.append(recall)
            #print ("user:%d precision:%.6f recall:%.6f" % (i + 1, precision, recall))

    print ("Precision:%.3f Recall:%.6f" % (np.mean(precisions), np.mean(recalls)))


if __name__ == "__main__":
    main()