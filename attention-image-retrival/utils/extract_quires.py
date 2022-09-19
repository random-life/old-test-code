import  os
import glob
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
import vgg16
def query_images(groundtruth_dir, img_dir, dataset):
    for f in glob.iglob(os.path.join(groundtruth_dir, '*_query.txt')):
        print (f)
        query_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '')
        img_name, x, y, w, h = open(f).read().strip().split(' ')
        if dataset == 'oxford':
            img_name = img_name.replace('oxc1_', '')
            img = imread(os.path.join(img_dir, '%s.jpg' % img_name))
        yield img, query_name


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='', help='dataset to extract queries for')
    parser.add_argument('--images', dest='images', type=str, default='data', help='directory containing image files')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='groundtruth', help='directory containing groundtruth files')
    parser.add_argument('--out', dest='out', type=str, default='pool5_queries', help='path to save output')
    args = parser.parse_args()
    images_dir = os.path.join(args.dataset,args.images)
    groundtruth_dir = os.path.join(args.dataset, args.groundtruth)

    out_dir = os.path.join(args.dataset, args.out)
    images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    sess = tf.Session()
    vgg = vgg16.vgg16(images, 'vgg16_weights.npz', sess)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(groundtruth_dir, images_dir)
    for img, name in query_images(groundtruth_dir, images_dir, dataset='oxford'):
        print(name)
        img = imresize(img, (224,224))
        rgb_img = img.reshape((1, 224, 224, 3))
        feed_dict = {images : rgb_img }
        pool = sess.run(vgg.pool5, feed_dict=feed_dict)
        feature = np.reshape(pool, [7, 7, 512])
        np.save(os.path.join(out_dir, '%s' % name), feature)

    sess.close()