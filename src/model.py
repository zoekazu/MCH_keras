#!/usr/bin/python
# -*- Coding: utf-8 -*-

from keras.layers import Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras_


def train_model():
    inputs = Input(shape=(30, 30, 1))

    f1 = Conv2D(filter=64, kernel_size=(5, 5), stride=(1, 1), padding='valid', data_format='channels_last',
                activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='Constant')(inputs)

    f2 = Conv2D(filter=32, kernel_size=(3, 3), stride=(1, 1), padding='valid', data_format='channels_last',
                activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='Constant')(f1)

    outputs = Conv2D(filter=4, kernel_size=(3, 3), stride=(1, 1), padding='valid', data_format='channels_last',
                     activation='liner', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='Constant')(f2)
    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=0.0001, decay=0.0001)
    model.compile(optimizer=adam, loss='mean_squared_error',
                  metrics=['mean_squared_error'], )


# def train_model():
#     # lrelu = LeakyReLU(alpha=0.1)
#     train_model = Sequential()
#     train_model.add(Conv2D(nb_filter=64, nb_row=5, nb_col=5, init='glorot_uniform',
#                            activation='relu', border_mode='valid', bias=True, input_shape=(30, 30, 1)))
#     train_model.add(Conv2D(nb_filter=32, nb_row=3, nb_col=3, init='glorot_uniform',
#                            activation='relu', border_mode='valid', bias=True))
#     # SRCNN.add(BatchNormalization())
#     train_model.add(Conv2D(nb_filter=4, nb_row=3, nb_col=3, init='glorot_uniform',
#                            activation='linear', border_mode='valid', bias=True))
#     adam = Adam(lr=0.0003)
#     train_model.compile(optimizer=adam, loss='mean_squared_error',
#                         metrics=['mean_squared_error'])
#     return train_model


# def predict_model():
#     # lrelu = LeakyReLU(alpha=0.1)
#     predict_model = Sequential()
#     predict_model.add(Conv2D(nb_filter=64, nb_row=5, nb_col=5, init='glorot_uniform',
#                              activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
#     predict_model.add(Conv2D(nb_filter=32, nb_row=3, nb_col=3, init='glorot_uniform',
#                              activation='relu', border_mode='valid', bias=True))
#     # SRCNN.add(BatchNormalization())
#     predict_model.add(Conv2D(nb_filter=4, nb_row=3, nb_col=3, init='glorot_uniform',
#                              activation='linear', border_mode='valid', bias=True))
#     return predict_model
