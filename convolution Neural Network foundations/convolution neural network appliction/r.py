import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

train_dataset = h5py.File('datasets/train_signs.h5', "r")
X_train_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
Y_train_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

test_dataset = h5py.File('datasets/test_signs.h5', "r")
X_test_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
Y_test_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

Y_train_orig = Y_train_orig.reshape((1, Y_train_orig.shape[0]))
Y_test_orig = Y_test_orig.reshape((1, Y_test_orig.shape[0]))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = (np.eye(6)[Y_train_orig.reshape(-1)].T).T

Y_test = (np.eye(6)[Y_test_orig.reshape(-1)].T).T



m = X_train.shape[0]
seed=3                # number of training examples
np.random.seed(seed)

# Step 1: Shuffle (X, Y)
permutation = list(np.random.permutation(m))
shuffled_X = X_train[permutation,:,:,:]
shuffled_Y = Y_train[permutation,:]
