import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tflearn
import numpy as np
hf = h5py.File('edges.h5')
X = hf['X']
Y = hf['Y']
X, Y = tflearn.data_utils.shuffle(X, Y)
hf =h5py.File('sketches.h5')
X2 = hf['X']
Y2 = hf['Y']
#X2, Y2 = tflearn.data_utils.shuffle(X2,Y2)
y1 = Y[0:32]
y2 = Y2[0:32]
y = (np.argmax(y1,1)==np.argmax(y2,1))
print(np.argmax(y1,1))
print(np.argmax(y2,1))
print(y.astype('float'))
plt.imshow(X2[2])
plt.show()
