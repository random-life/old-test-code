import os
import sys
import glob
from functools import partial
from tempfile import NamedTemporaryFile
import numpy as np
from skimage import io
from crow import    run_feature_processing_pipeline,compute_crow_channel_weight,compute_crow_spatial_weight, apply_crow_aggregation,apply_ucrow_aggregation, apply_max_aggregation,normalize
from evaluate import load_features, get_nn, load_and_aggregation_features


data, image_names = load_and_aggregation_features('shoefeatures',apply_crow_aggregation)

data = normalize(np.vstack(data), copy=False)

Q = np.load('shoefeatures/shoe_001.npy')
Q = apply_crow_aggregation(Q)
Q = normalize(Q, copy=False)
inds, dists = get_nn(Q, data)
for i in inds:
    print(image_names[i])