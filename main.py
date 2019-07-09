#!/usr/bin/python
# -*- Coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from generate_dataset import read_training_data
import numpy as np
import math

from src.separate_composit import composit_output
from src import psnr
from src import utils
from src.image_processing import modcrop, psnr
from model import train_model, predict_model


def train():
    srcnn_model = train_model()
    print(srcnn_model.summary())
    data, label = read_training_data("./hdf5/train.h5")
    val_data, val_label = read_training_data("./hdf5/test.h5")

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, nb_epoch=200, verbose=0)
    # srcnn_model.load_weights("m_model_adam.h5")


def predict():
    _shave = 8
    srcnn_model = predict_model()
    srcnn_model.load_weights("SRCNN_check.h5")
    IMG_NAME = "dataset/Test/Set14/lenna.bmp"
    OUTPUT_NAME = "output.bmp"

    import cv2
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    Y_img = modcrop(img, 2)

    shape = Y_img.shape

    Y_img_s = cv2.resize(Y_img, (shape[1] // 2,
                                 shape[0] // 2), cv2.INTER_CUBIC)

    Y = np.zeros((1, Y_img_s.shape[0], Y_img_s.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img_s.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre = composit_output(pre)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    cv2.imwrite(OUTPUT_NAME, pre)

    ref_img = Y_img[_shave-1: -(_shave+1), _shave-1: -(_shave+1)]

    # psnr calculation:
    print('pnsr:', psnr(ref_img, pre))


if __name__ == "__main__":
    # generate_dataset()
    # train()
    predict()
