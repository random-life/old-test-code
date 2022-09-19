import os
import glob
import  numpy as np
import cv2
import tensorflow as tf
import scipy
from functools import partial
from vgg import vgg16
from crow import  normalize, save_spatial_weights_as_jpg,compute_crow_spatial_weight,compute_crow_channel_weight
from evaluate import load_features

num_largest = 7
def save_spatial_weights_as_jpg(S, path='.', filename='crow_sw', size=None):
    img = scipy.misc.toimage(S)
    if size is None:
        size = (S.shape[1] * 32, S.shape[0] * 32)
    img = img.resize(size, cv2.INTER_CUBIC)
    img.save(os.path.join(path, '%s.jpg' % str(filename)))

for X , name in load_features('shoefeatures'):
    S = compute_crow_spatial_weight(X)
    im2 = cv2.imread(os.path.join('shoes','%s.jpg' % str(name)))
    d = np.float32(im2)
    d = d.transpose((2, 0, 1))
    indices = (-S).argpartition(num_largest, axis=None)[:num_largest]
    xs, ys = np.unravel_index(indices, S.shape)
    save_spatial_weights_as_jpg(S, 'shoesw', filename=name, size=(d.shape[2], d.shape[1]))


