#!/usr/bin/python
# -*- coding: utf-8 -*-
""" @ref    https://6chaoran.github.io/data-story/deepfm-for-recommendation/
    @paper  https://arxiv.org/pdf/1703.04247.pdf
    @code   https://github.com/Leavingseason/xDeepFM
    other deepfm
    https://github.com/ChenglongChen/tensorflow-DeepFM # theano
    https://github.com/facebookresearch/dlrm # pytorch
"""
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd

__all__ = ["DeepFM"]

class _deepfm(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def define_input_layers():
        # numeric features
        # fea3_input = Input((1,), name='input_fea3')
        # num_inputs = [fea3_input]
        # single level categorical features
        uid_input = Input((1,), name='input_uid')
        mid_input = Input((1,), name='input_mid')
        cat_sl_inputs = [uid_input, mid_input]

        # multi level categorical features (with 3 genres at most)
        # genre_input = Input((3,), name='input_genre')
        # cat_ml_inputs = [genre_input]

        inputs = cat_sl_inputs

        return inputs

    @staticmethod
    def Tensor_Mean_Pooling(name='mean_pooling', keepdims=False):
        return Lambda(lambda x: K.mean(x, axis=1, keepdims=keepdims), name=name)

    def fm_1d(inputs, n_uid, n_mid):
        uid_input, mid_input = inputs

        # all tensors are reshape to (None, 1)
        # num_dense_1d = [Dense(1, name='num_dense_1d_fea4')(fea3_input)]
        cat_sl_embed_1d = [Embedding(n_uid + 1, 1, name='cat_embed_1d_uid')(uid_input),
                           Embedding(n_mid + 1, 1, name='cat_embed_1d_mid')(mid_input)]
        # cat_ml_embed_1d = [Embedding(n_genre + 1, 1, mask_zero=True, name='cat_embed_1d_genre')(genre_input)]

        cat_sl_embed_1d = [Reshape((1,))(i) for i in cat_sl_embed_1d]
        # cat_ml_embed_1d = [__class__.Tensor_Mean_Pooling(name='embed_1d_mean')(i) for i in cat_ml_embed_1d]

        # add all tensors
        y_fm_1d = Add(name='fm_1d_output')(cat_sl_embed_1d)

        return y_fm_1d

    @staticmethod
    def fm_2d(inputs, n_uid, n_mid, k):
        uid_input, mid_input = inputs

        # num_dense_2d = [Dense(k, name='num_dense_2d_fea3')(fea3_input)]  # shape (None, k)
        # num_dense_2d = [Reshape((1, k))(i) for i in num_dense_2d]  # shape (None, 1, k)

        cat_sl_embed_2d = [Embedding(n_uid + 1, k, name='cat_embed_2d_uid')(uid_input),
                           Embedding(n_mid + 1, k, name='cat_embed_2d_mid')(mid_input)]  # shape (None, 1, k)

        # cat_ml_embed_2d = [Embedding(n_genre + 1, k, name='cat_embed_2d_genre')(genre_input)]  # shape (None, 3, k)
        # cat_ml_embed_2d = [__class__.Tensor_Mean_Pooling(name='cat_embed_2d_genure_mean', keepdims=True)(i) for i in
        #                    cat_ml_embed_2d]  # shape (None, 1, k)

        # concatenate all 2d embed layers => (None, ?, k)
        embed_2d = Concatenate(axis=1, name='concat_embed_2d')(cat_sl_embed_2d)

        # calcuate the interactions by simplication
        # sum of (x1*x2) = sum of (0.5*[(xi)^2 - (xi^2)])
        tensor_sum = Lambda(lambda x: K.sum(x, axis=1), name='sum_of_tensors')
        tensor_square = Lambda(lambda x: K.square(x), name='square_of_tensors')

        sum_of_embed = tensor_sum(embed_2d)
        square_of_embed = tensor_square(embed_2d)

        square_of_sum = Multiply()([sum_of_embed, sum_of_embed])
        sum_of_square = tensor_sum(square_of_embed)

        sub = Subtract()([square_of_sum, sum_of_square])
        sub = Lambda(lambda x: x * 0.5)(sub)
        y_fm_2d = Reshape((1,), name='fm_2d_output')(tensor_sum(sub))

        return y_fm_2d, embed_2d

    @staticmethod
    def deep_part(embed_2d, dnn_dim, dnn_dr):
        # flat embed layers from 3D to 2D tensors
        y_dnn = Flatten(name='flat_embed_2d')(embed_2d)
        for h in dnn_dim:
            y_dnn = Dropout(dnn_dr)(y_dnn)
            y_dnn = Dense(h, activation='relu')(y_dnn)
        y_dnn = Dense(1, activation='relu', name='deep_output')(y_dnn)

        return y_dnn

class _util(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def encode_sequence(data):
        assert isinstance(data, (np.ndarray, list))
        assert (len(data) > 0)
        assert isinstance(data[0], (np.int64, np.int32, np.int16))
        u_data = np.unique(data)
        u_data = sorted(u_data)
        iid = range(len(data))
        iid2data = dict(zip(iid, u_data))
        data2iid = dict(zip(u_data, iid))
        return iid2data, data2iid

    @staticmethod
    def df2xy(ratings):
        assert isinstance(ratings, pd.DataFrame)
        assert ("userid" in ratings.columns)
        assert ("itemid" in ratings.columns)
        x = [#ratings.user_fea3.values,
             ratings.userid.values,
             ratings.itemid.values,
             #np.concatenate(ratings.movie_genre.values).reshape(-1, 3)
             ]
        y = ratings.rating.values
        return x, y

class DeepFM(_deepfm, _util):
    def __init__(self, n_uid=None, n_mid=None,
                 k=20, dnn_dim=[64,64], dnn_dr=0.5,
                 filepath="../data/deepfm_weights.h5"):
        super().__init__()

        inputs = self.define_input_layers()

        y_fm_1d = __class__.fm_1d(inputs, n_uid, n_mid)
        y_fm_2d, embed_2d = __class__.fm_2d(inputs, n_uid, n_mid, k)
        y_dnn = __class__.deep_part(embed_2d, dnn_dim, dnn_dr)

        # combined deep and fm parts
        y = Concatenate()([y_fm_1d, y_fm_2d, y_dnn])
        y = Dense(1, name='deepfm_output')(y)

        # fm_model_1d = Model(inputs, y_fm_1d)
        # fm_model_2d = Model(inputs, y_fm_2d)
        # deep_model = Model(inputs, y_dnn)
        deep_fm_model = Model(inputs, y)
        self.model_ = deep_fm_model

        self.model_.compile(loss='MSE', optimizer='adam')

        early_stop = EarlyStopping(monitor='val_loss', patience=3)

        """ @ref https://machinelearningmastery.com/check-point-deep-learning-models-keras/ """
        model_ckp = ModelCheckpoint(filepath=filepath,
                                    monitor='val_loss',
                                    save_weights_only=True,
                                    save_best_only=True)

        self.callbacks_ = [model_ckp, early_stop]

    def fit(self, train_x, train_y, epochs=30, batch_size=2048, validation_split=0.1):
        return self.model_.fit(train_x, train_y,
                epochs=epochs, batch_size=batch_size,
                validation_split=validation_split,
                callbacks=self.callbacks_)

    def evaluate(self, X, y, verbose=1):
        results = self.model_.evaluate(X, y, verbose=verbose)
        print("DeepFM.evaluate test_loss: {0}".format(results))
        return results

    def predict(self, X, verbose=1):
        predicted = self.model_.predict(X, verbose=verbose)
        return predicted


