import numpy as np
import scipy
import cv2
from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA
import os
def floodfill(matrix, x, y, l, num, labels):

    if matrix[x][y] == 1 and labels[x][y]==0 :
        matrix[x][y] = l
        num[l] += 1
        labels[x][y] = 1
        if x > 0:
            floodfill(matrix, x - 1, y, l, num, labels)
        if x < len(matrix[y]) - 1:
            floodfill(matrix, x + 1, y, l, num, labels)
        if y > 0:
            floodfill(matrix, x, y - 1, l, num, labels)
        if y < len(matrix) - 1:
            floodfill(matrix, x, y + 1, l, num, labels)
    else :
        return 0


def scda (X):
    M = np.zeros((X.shape[1],X.shape[2]))
    S = X.sum(axis=(0))
    threshod = S.mean()
    M = np.where(S>=threshod, 1, 0)
    num = np.zeros((M.shape[0]*M.shape[1]))
    l = 2
    labels = np.zeros((X.shape[1],X.shape[2]))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            floodfill(M, i, j, l, num, labels)
            l += 1
    l_max = np.argmax(num)
    M = np.where(M == l_max, 1, 0)
    return M, num[l_max]


X = np.load(os.path.join('shoefeatures','shoe_001.npy'))
M, num= scda(X)
X = np.where(M==1,X,0)
print (X.shape)
print (num)

img = scipy.misc.toimage(M)

size = (M.shape[1] * 32, M.shape[0] * 32)
img = img.resize(size, cv2.INTER_CUBIC)
img.save(os.path.join('shoesw', 'test.jpg'  ))
