#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @brief  Recommendation algorithm using optimized Matrix Factorization in TensorFlow
    @author <ariel kalingking> akalingking@gmail.com """
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from metrics import precision_at_k, recall_at_k
import metrics2
if sys.version_info[0] < 3:
    range = xrange

def main():
    session = tf.Session()
    normalized_on = False
    k = 100

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
    ratings = ratings.astype(np.float64)

    """ Construct training data """
    # train_size = 0.7
    ratings_train_ = ratings#.loc[:int(n_users*train_size), :int(n_items*train_size)]
    users = ratings_train_.index.values
    items = ratings_train_.columns.values
    n_users = len(users)
    n_items = len(items)
    temp = ratings_train_.copy()
    rating_mean = temp.replace(0, np.NaN).mean().mean()
    rating_mean = 3.5 if rating_mean > 3.5 else rating_mean

    print ("Training data:")
    print ("Shape: %s" % str(ratings_train_.shape))
    print ("n_users: %d" % n_users)
    print ("n_items: %d" % n_items)
    print ("rating mean: %.5f" % rating_mean)

    user_indices = [x for x in range(n_users)]
    item_indices = [x for x in range(n_items)]

    print ("Max userid train: ", np.max(users))
    print ("Max itemid train", np.max(items))
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
    lr = tf.constant(.0001)
    global_step = tf.Variable(0, trainable=False)
    decaying_learning_rate = tf.train.exponential_decay(lr, global_step, 10000, .96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(decaying_learning_rate).minimize(cost, global_step=global_step)

    """ Run """
    init = tf.global_variables_initializer()
    session.run(init)

    print ("Running stochastic gradient descent..")
    epoch = 500
    for i in range(epoch):
        session.run(optimizer)
        if i%10 == 0 or i == epoch-1:
            diff_op_train = tf.subtract(ratings.astype(np.float32), R_masked)
            diff_op_train_squared = tf.square(diff_op_train)
            se = tf.reduce_sum(diff_op_train_squared)
            mse = tf.divide(se, n_users*n_items)
            rmse = tf.sqrt(mse)
            print("Train iter: %d MSE: %.5f loss: %.5f" % (i, session.run(rmse), session.run(cost)))

    R_hat = R_.eval(session=session)
    ratings_csr = sparse.csr_matrix(ratings)
    precision = metrics2.precision_at_k(ratings_csr, R_hat, k=k)
    recall = metrics2.recall_at_k(ratings_csr, R_hat, k=k)
    print("Precision:%.3f%% Recall:%.3f%%" % (precision * 100, recall * 100))


if __name__ == "__main__":
    main()