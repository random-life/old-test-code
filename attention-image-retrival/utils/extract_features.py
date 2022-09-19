#from vgg import vgg16,utils
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy
import os
from scipy.misc import imread, imresize
import vgg16
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--images', dest='images',type=str,  default="shoes", nargs='+' ,
                        help='glob pattern to image data')
    parser.add_argument('--out', dest='out', type=str, default='shoefeatures', help='path to save output')

    args = parser.parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out)

sess =tf.Session()
images = tf.placeholder("float", [None, None, None, 3])
vgg = vgg16.vgg16(images, 'vgg16_weights.npz', sess)
for root,dir,f in os.walk(args.images):
    for path in f:
        path = os.path.join(root,path)
        print(path)
        img = imread(path)
        if img is None:
            print(path)
            continue
        #img = imresize(img,(224,224))
        rgb_img = img.reshape((1, img.shape[0], img.shape[1], 3))
           # print(batch.shape)
        feed_dict = {images: rgb_img}
        pool = sess.run(vgg.pool5, feed_dict=feed_dict)
        pool = np.reshape(pool,(pool.shape[1], pool.shape[2], pool.shape[3]))

        pool = pool.transpose((2, 0, 1))
        print(pool.shape)
        #feature = np.reshape(pool, [7, 7, 512])
        filename = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(args.out, filename), pool)