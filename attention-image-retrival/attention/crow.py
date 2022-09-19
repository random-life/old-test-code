import numpy as np
import scipy
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA
import os


def compute_crow_spatial_weight(X, a=2, b=2):
    S = X.sum(axis=0)
    z = (S**a).sum()**(1./a)
    return  (S / z)**(1./b) if b != 1 else(S / z)
def  compute_crow_channel_weight(X):
    K, w, h = X.shape
    area = float(w * h)
    none_zeros = np.zeros(K, dtype=np.float32)
    for i, x in enumerate(X):
        none_zeros[i] = np.count_nonzero(x)/area

        nzsum = none_zeros.sum()
    for i, d in enumerate(none_zeros):
       none_zeros[i] = np.log(nzsum / d) if d > 0. else 0.
    return none_zeros

def apply_crow_aggregation(X):

    S = compute_crow_spatial_weight(X)
    C = compute_crow_channel_weight(X)
    X = S * X
    X = X.sum(axis=(1,2))

    return X * C
def apply_ucrow_aggregation(X):
    return X.sum(axis=(1 ,2))
def apply_max_aggregation(X):
    return X.max(axis=(1, 2))
def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)
def run_feature_processing_pipeline(features, d=128, whiten=True,copy=False,params=None):
    features = normalize(features, copy=copy)
    if params:
        pca = params['pca']
        features = pca.transform(features)
    else:
        pca =PCA(n_components=d, copy=copy, whiten=whiten)
        features = pca.fit_transform(features)
        params = {'pca' : pca}

    features = normalize(features, copy=copy)

    return features, params

def save_spatial_weights_as_jpg(S, path='sw', filename='crow_sw', size=None):
    img = scipy.misc.toimage(S)
    #size =(S.shape[0] * 32, S.shape[1] * 32)
    if size is not None:
        img =img.resize(size)
    img.save(os.path.join(path, '%s.jpg' % str(filename)))
