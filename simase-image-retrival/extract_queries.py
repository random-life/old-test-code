import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
from sklearn.preprocessing import normalize
import h5py
import tensorflow as tf
import numpy as np
import cv2
import scipy
input  = tflearn.input_data(shape=[None,224,224,3], name='input')
with tf.variable_scope('siamese') as scope:
	x = tflearn.conv_2d(input, 64, 3, activation='relu' ,scope='conv1_1')
	x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
	x = tflearn.max_pool_2d(x, 2, strides=2)
	'''kernel = tf.Variable(tf.truncated_normal([1,1,64,64],dtype=tf.float32,stddev=1e-1),name='attention_conv1_weights')
	conv = tf.nn.conv2d(x,kernel,[1,1,1,1],padding='SAME')
	conv = tf.nn.relu(conv)
	kernel2 = tf.Variable(tf.truncated_normal([1,1,64,1],dtype=tf.float32,stddev=1e-1), name='attention_conv2_weights')
	conv2 = tf.nn.conv2d(conv,kernel2,[1,1,1,1], padding='SAME')
	attention_score = tf.nn.relu(conv2)
	x = attention_score * x
	c = tflearn.avg_pool_2d(x, 112)
	c = tflearn.flatten(c)
	c1 = tflearn.fully_connected(c, 4, activation='relu')
	c2 = tflearn.fully_connected(c1, 64, activation='sigmoid')
	c2 = tf.expand_dims(c2, 1)
	c2 = tf.expand_dims(c2, 2)
	x = x * c2'''
	#attention_c = tf.get_variable('attention_c', shape=[64],initializer=tf.random_normal_initializer())
	#x = x * attention_c


	x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
	x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
	x = tflearn.max_pool_2d(x, 2, strides=2)
	kernel = tf.get_variable(name='attention_conv3_weights',initializer=tf.truncated_normal([1,1,128,128],dtype=tf.float32,stddev=1e-1))
	conv = tf.nn.conv2d(x, kernel,[1,1,1,1],padding='SAME')
	conv = tf.nn.relu(conv)
	kernel2 = tf.get_variable(name='attention_conv4_weights', initializer=tf.truncated_normal([1,1,128,1],dtype=tf.float32, stddev=1e-1))
	conv2 = tf.nn.conv2d(conv,kernel2,[1,1,1,1], padding='SAME')
	attention_score2= tf.nn.relu(conv2)
	x = x * attention_score2
	c = tflearn.avg_pool_2d(x,56)

	c = tflearn.flatten(c)
	c3 = tflearn.fully_connected(c,8,activation='relu',scope='c3')
	c4 = tflearn.fully_connected(c3,128, activation='sigmoid',scope='c4')
	c4 = tf.expand_dims(c4,1)
	c4 = tf.expand_dims(c4, 2)
	x = x * c4

	x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
	x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
	x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
	x = tflearn.max_pool_2d(x, 2, strides=2)
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
	x = tflearn.max_pool_2d(x, 2, strides=2)
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
	x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
	x = tflearn.max_pool_2d(x, 2, strides=2)
	'''kernel = tf.Variable(tf.truncated_normal([1,1,512,512],dtype=tf.float32,stddev=1e-1),name='attention_1_weights')
	conv = tf.nn.conv2d(x, kernel,[1,1,1,1],padding='SAME')
	f_conv1_attention = tf.nn.relu(conv)

	kernel2 = tf.Variable(tf.truncated_normal([1,1,512,1],dtype=tf.float32,stddev=1e-1),name='attention_2_weights')
	conv2 = tf.nn.conv2d(f_conv1_attention, kernel2,[1,1,1,1],padding='SAME')
	attention_score = tf.nn.relu(conv2, name='score')
	attention_feat = x * attention_score
	#attention_feat = tf.expand_dims(tf.expand_dims(attention_feat, 1), 2)'''


	#attention_c = tf.get_variable('attention_c_final', shape=[512],initializer=tf.random_normal_initializer())
	#x = x * attention_c
	#kernel3 = tf.Variable(tf.truncated_normal([1,1,512,num_class],dtype=tf.float32,
	#                                         stddev=1e-1),name='m2d_weights')
	#conv3 = tf.nn.conv2d(x, kernel3,[1,1,1,1],padding='SAME')
	#logits = tf.squeeze(conv3,[1,2])
	#x = tf.nn.softmax(logits)

	x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
	x = tflearn.dropout(x, 0.5, name='dropout1')
	fc7 = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')

import matplotlib.pyplot as plt
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,'model')
    for root, dir, f in os.walk('source_edge'):
        for p in f :
            path = os.path.join(root,p)
            print(path)
            name = path.replace('source_edge','score-np')
            image = cv2.imread(path)
            print(name)
            image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
            image = image.reshape((1,224,224,3))
            score = sess.run(attention_score2,feed_dict={input:image})
	    
            score = np.squeeze(score)
            np.save(name,score)
            #show = scipy.misc.toimage(score)
            #show = show.resize((224,224))
            #show.save(name)
            #fig, ax = plt.subplots(1)
            #implot = ax.imshow(np.squeeze(image))
            #heatmap = ax.pcolor(show, alpha=0.1)
            #plt.axis('off')
            #plt.savefig(name)
            '''name = path.replace('sketch_edge','sketch-features').replace('.jpg','')
            image = cv2.imread(path)
            print(name)
            image = cv2.resize(image,(224,224))
            image = image / 255.0
            image = image.reshape((1,image.shape[0],image.shape[1],3))
            feature = sess.run(fc7,feed_dict={input:image})
            #feature = np.sum(feature, axis=(1,2))
            #feature = normalize(feature.reshape(1,-1))
            feature = np.squeeze(feature)
            print(feature.shape)
            #feature = np.maximum(feature,0)
            np.save(os.path.join('sketch-retrival',name),feature)'''

