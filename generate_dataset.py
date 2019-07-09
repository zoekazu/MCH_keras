#!/usr/bin/env python3
# -*- Coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import cv2
import h5py

from src.utils import confirm_make_folder
from src.read_dir_imags import ImgInDirAsY
from src.separate_composit import separate_label

TRAIN_PATH = "./dataset/Train/"
TEST_PATH = "./dataset/Test/Set14/"

PATCH_SIZE = (32)
STRIDE_SIZE = (16)
INPUT_SIZE = (30)
LABEL_SIZE = (22)
SCALE = (2)
SHAVE_SIZE = abs(INPUT_SIZE - LABEL_SIZE) // 2


def prepare_dataset(_path):
    img_files = ImgInDirAsY(_path)

    data = []
    label = []

    for hr_img in img_files.read_files():

        hr_img = modcrop(hr_img, SCALE)
        _wid, _hei = hr_img.shape

        for y in range(0, _hei - INPUT_SIZE*SCALE, STRIDE_SIZE):
            for x in range(0, _wid - INPUT_SIZE*SCALE, STRIDE_SIZE):
                hr_patch = hr_img[x: x + INPUT_SIZE *
                                  SCALE, y: y + INPUT_SIZE*SCALE]
                lr_patch = cv2.resize(
                    hr_patch, dsize=None, fx=1/SCALE, fy=1/SCALE, interpolation=cv2.INTER_CUBIC)

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                hr_patch = hr_patch
                hr_patch_separete = separate_label(hr_patch, SCALE)[:, SHAVE_SIZE-1: -(SHAVE_SIZE+1),
                                                                    SHAVE_SIZE-1: -(SHAVE_SIZE+1)]

                lr = np.zeros(
                    (1, INPUT_SIZE, INPUT_SIZE), dtype=np.double)
                hr = np.zeros(
                    (SCALE ** 2, LABEL_SIZE, LABEL_SIZE), dtype=np.double)

                lr[0, :, :] = lr_patch
                hr[:, :, :] = hr_patch_separete

                data.append(lr)
                label.append(hr)

    data = np.array(data, dtype=float)
    label = np.array(label, dtype=float)
    return data, label


def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    data = data.astype(np.float32)
    labels = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)
        h.create_dataset('label', data=labels, shape=labels.shape)

    scan_hdf5(output_filename)


def scan_hdf5(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            elif isinstance(v, h5py.Group) and recursive:
                scan_node(v, tabs=tabs + tab_step)
    with h5py.File(path, 'r') as f:
        scan_node(f)


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


def modcrop(img, scale):
    if img.ndim == 3:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1], :]
    else:
        img_size = np.array(img.shape[0:2])
        img_size = img_size - np.mod(img_size, scale)
        out_img = img[0:img_size[0], 0: img_size[1]]

    return out_img


if __name__ == "__main__":
    confirm_make_folder('./hdf5')
    data, label = prepare_dataset(TRAIN_PATH)
    write_hdf5(data, label, "./hdf5/train.h5")
    data, label = prepare_dataset(TEST_PATH)
    write_hdf5(data, label, "./hdf5/test.h5")