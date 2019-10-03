#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
sys.path.append("../")
import metrics
from deepfm import DeepFM

class TestDeepFM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(80000)
        datafile = "../data/ml-100k/u.data"
        ratings = pd.read_csv(datafile, sep='\t', names=["userid", "itemid", "rating", "timestamp"])

        """ Convert rating data to user x movie matrix format"""
        # data = data.sort_values(by=["userid", "itemid"])
        # ratings = pd.pivot_table(data, values="rating", index="userid", columns="itemid")
        # ratings.fillna(0, inplace=True)

        users = np.unique(ratings.userid.values)
        items = np.unique(ratings.itemid.values)
        n_users = len(users)
        n_items = len(items)
        print("n_users:%d n_items=%d" % (n_users, n_items))
        cls._ratings = ratings
        cls._n_users = n_users
        cls._n_items = n_items
        cls._k = 100
        cls._epochs = 10

    @classmethod
    def tearDownClass(cls):
        pass

    def deepfm_test(self):
        train_x, train_y = DeepFM.df2xy(self._ratings)
        #test_x, test_y = DeepFM.df2xy(self.test_data_)

        params = {
            'n_uid': self._ratings.userid.max(),
            'n_mid': self._ratings.itemid.max(),
            # 'n_genre': self.n_genre_,
            'k': self._k,
            'dnn_dim': [64, 64],
            'dnn_dr': 0.5,
            'filepath': '../data/deepfm_weights.h5'
        }

        """ train """
        model = DeepFM(**params)
        train_history = model.fit(train_x,
                                  train_y,
                                  epochs=self._epochs,
                                  batch_size=2048,
                                  validation_split=0.1)

        history = pd.DataFrame(train_history.history)
        history.plot()
        plt.savefig("../data//history.png")

        """ test """
        results = model.evaluate(train_x, train_y)
        print("Validate result:{0}".format(results))

        """ predict """
        y_hat = model.predict(train_x)

        print(np.shape(y_hat))
        # print(np.shape(test_y))

        """ Run Recall and Precision Metrics """
        n_users = np.max(self._ratings.userid.values) + 1
        n_items = np.max(self._ratings.itemid.values) + 1
        print("n_users={0} n_items={1}".format(n_users, n_items))

        # Convert to sparse matrix to run standard metrics
        sparse_train = sparse.coo_matrix((self._ratings.rating.values,
                                          (self._ratings.userid.values, self._ratings.itemid.values)),
                                         shape=(n_users, n_items))

        # sparse_test = sparse.coo_matrix((self.test_data_.rating.values, \
        #                                  (self.test_data_.uid.values, self.test_data_.mid.values)), \
        #                                 shape=(n_users, n_items))
        # pd.DataFrame(data=sparse_test.tocsr().todense().A).to_csv("./testdata.csv")

        # test_prediced
        test_predicted = self._ratings.copy()
        test_predicted.rating = np.round(y_hat)

        sparse_predicted = sparse.coo_matrix((test_predicted.rating.values, \
                                              (test_predicted.userid.values, test_predicted.itemid.values)), \
                                             shape=(n_users, n_items))

        sparse_train_1up = sparse_train.multiply(sparse_train >= 1)
        # sparse_test_1up = sparse_test.multiply(sparse_test >= 1)

        predicted_arr = sparse_predicted.tocsr().todense().A
        predicted_ranks = metrics.rank_matrix(predicted_arr)

        precision_ = metrics.precision_at_k(predicted_ranks, sparse_train, k=self._k)
        recall_ = metrics.recall_at_k(predicted_ranks,  sparse_train, k=self._k)

        print("{0}.xdeepfm_test train precision={1:.4f}% recall={2:.4f}% @k={3}".format(
            __class__.__name__, precision_ * 100, recall_ * 100, self._k))
