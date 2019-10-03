#!/usr/bin/python
# -*- encoding: utf-8 -*-
import unittest
import numpy as np
from tensorrec import TensorRec
from tensorrec import util, eval
import metrics

class TestMetric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()

    def metric_test(self):
        """ uses tensorrec eval as benchmark for rating performance of various reco algorithms """
        k = 10
        latent_factor = 10
        n_users = 10
        n_items = 12

        interactions, user_features, item_features = util.generate_dummy_data_with_indicator (num_users=n_users, num_items=n_items, interaction_density=.5)
        print ("interactiosn shape={}".format( np.shape(interactions) ))
        print ("user features shape={}".format( np.shape(user_features.toarray()) ))
        print ("item features shape={}".format( np.shape(item_features.toarray()) ))

        model = TensorRec(n_components=latent_factor)

        model.fit(interactions, user_features, item_features, epochs=19)

        ranks = model.predict_rank(user_features=user_features, item_features=item_features)

        print ("Ranks shape={}".format(np.shape(ranks)))

        self.assertTrue(np.shape(interactions) ==  np.shape(ranks))

        tr_recall_result = eval.recall_at_k(predicted_ranks=ranks, test_interactions=interactions, k=k, preserve_rows=False)
        # print (tr_recall_result.mean())

        tr_precision_result = eval.precision_at_k(predicted_ranks=ranks, test_interactions=interactions, k=k, preserve_rows=False)
        # print(tr_precision_result.mean())

        # we need csr for interactions data
        interactions_ = interactions.tocsr()
        recall_result = metrics.recall_at_k(ranks, interactions_, k=k, preserve_rows=False)
        # print(recall_result.mean())

        precision_result = metrics.precision_at_k(ranks, interactions_, k=k, preserve_rows=False)
        # print (precision_result.mean())

        self.assertTrue (tr_recall_result.mean() == recall_result.mean())
        self.assertTrue (tr_precision_result.mean() == precision_result.mean())

