import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
import h5py
import tensorflow as tf
import numpy as np
from  tflearn.datasets import oxflower17
#X, Y = oxflower17.load_data(one_hot=True)
class siamese:

	
	def __init__(self):
		self.x1 = tf.placeholder(tf.float32,[None,224,224,3])
		self.x2 = tf.placeholder(tf.float32,[None,224,224,3])
		with tf.variable_scope("siamese") as scope:
			self.o1 = self.network(self.x1)
			scope.reuse_variables()
			self.o2 = self.network(self.x2)
		self.y_ = tf.placeholder(tf.float32, [None])
		self.loss = self.loss_with_spring()

	



	def network(self, input):
		datadict = np.load('vgg16_weights.npz')
		#print(datadict.keys())
		#print(datadict['conv1_1_W'].shape)
		conv1_1 = tf.constant(datadict['conv1_1_W'])
		bias1_1 = tf.constant(datadict['conv1_1_b'])
		conv1_2 = tf.constant(datadict['conv1_2_W'])
		bias1_2 = tf.constant(datadict['conv1_1_b'])
		conv2_1 = tf.constant(datadict['conv2_1_W'])
		bias2_1 = tf.constant(datadict['conv2_1_b'])
		conv2_2 = tf.constant(datadict['conv2_2_W'])
		bias2_2 = tf.constant(datadict['conv2_2_b'])
		conv3_1 = tf.constant(datadict['conv3_1_W'])
		bias3_1 = tf.constant(datadict['conv3_1_b'])
		conv3_2 = tf.constant(datadict['conv3_2_W'])
		bias3_2 = tf.constant(datadict['conv3_2_b'])
		conv3_3 = tf.constant(datadict['conv3_3_W'])
		bias3_3 = tf.constant(datadict['conv3_3_b'])
		conv4_1 = tf.constant(datadict['conv4_1_W'])
		bias4_1 = tf.constant(datadict['conv4_1_b'])
		conv4_2 = tf.constant(datadict['conv4_2_W'])
		bias4_2 = tf.constant(datadict['conv4_2_b'])
		conv4_3 = tf.constant(datadict['conv4_3_W'])
		bias4_3 = tf.constant(datadict['conv4_3_b'])
		conv5_1 = tf.constant(datadict['conv5_1_W'])
		bias5_1 = tf.constant(datadict['conv5_1_b'])
		conv5_2 = tf.constant(datadict['conv5_2_W'])
		bias5_2 = tf.constant(datadict['conv5_2_b'])
		conv5_3 = tf.constant(datadict['conv5_3_W'])
		bias5_3 = tf.constant(datadict['conv5_3_b'])
		fc6_w = tf.constant(datadict['fc6_W'])
		fc6_b = tf.constant(datadict['fc6_b'])
		fc7_w = tf.constant(datadict['fc7_W'])
		fc7_b = tf.constant(datadict['fc7_b'])
		x = tflearn.conv_2d(input, 64, 3, activation='relu', weights_init=conv1_1,bias_init=bias1_1, scope='conv1_1')
		x = tflearn.conv_2d(x, 64, 3, activation='relu',weights_init=conv1_2,bias_init=bias1_2, scope='conv1_2')
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
		x = tflearn.conv_2d(x, 128, 3, activation='relu',weights_init=conv2_1,bias_init=bias2_1, scope='conv2_1')
		x = tflearn.conv_2d(x, 128, 3, activation='relu',weights_init=conv2_2,bias_init=bias2_2, scope='conv2_2')
		x = tflearn.max_pool_2d(x, 2, strides=2)
		'''kernel = tf.get_variable(name='attention_conv3_weights',initializer=tf.truncated_normal([1,1,128,128],dtype=tf.float32,stddev=1e-1))
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
		x = x * c4'''

		x = tflearn.conv_2d(x, 256, 3, activation='relu',weights_init=conv3_1,bias_init=bias3_1, scope='conv3_1')
		x = tflearn.conv_2d(x, 256, 3, activation='relu',weights_init=conv3_2,bias_init=bias3_2, scope='conv3_2')
		x = tflearn.conv_2d(x, 256, 3, activation='relu',weights_init=conv3_3,bias_init=bias3_3, scope='conv3_3')
		x = tflearn.max_pool_2d(x, 2, strides=2)
		x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv4_1,bias_init=bias4_1, scope='conv4_1')
		x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv4_2,bias_init=bias4_2, scope='conv4_2')
		x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv4_3,bias_init=bias4_3, scope='conv4_3')
		x = tflearn.max_pool_2d(x, 2, strides=2)
		x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv5_1,bias_init=bias5_1, scope='conv5_1')
		x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv5_2,bias_init=bias5_2, scope='conv5_2')
		x = tflearn.conv_2d(x, 512, 3, activation='relu',weights_init=conv5_3,bias_init=bias5_3, scope='conv5_3')
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

		x = tflearn.fully_connected(x, 4096, activation='relu',weights_init=fc6_w,bias_init=fc6_b, scope='fc6')
		x = tflearn.dropout(x, 0.5, name='dropout1')
		x = tflearn.fully_connected(x, 4096, activation='relu',weights_init=fc7_w,bias_init=fc7_b, scope='fc7')

		return x

	def loss_with_spring(self):
		margin = 5.0
		labels_t = self.y_
		labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
		eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
		eucd2 = tf.reduce_sum(eucd2, 1)
		eucd = tf.sqrt(eucd2+1e-6, name="eucd")
		C = tf.constant(margin, name="C")
		# yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
		pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
		# neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
		# neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
		neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
		losses = tf.add(pos, neg, name="losses")
		loss = tf.reduce_mean(losses, name="loss")
		return loss

	def loss_with_step(self):
		margin = 5.0
		labels_t = self.y_
		labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
		eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
		eucd2 = tf.reduce_sum(eucd2, 1)
		eucd = tf.sqrt(eucd2+1e-6, name="eucd")
		C = tf.constant(margin, name="C")
		pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
		neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
		losses = tf.add(pos, neg, name="losses")
		loss = tf.reduce_mean(losses, name="loss")
		return loss

#img_prep = ImagePreprocessing()
#img_prep.add_featurewise_zero_center(mean=[123.68,116.779,103.939],per_channel=True)
#x = tflearn.input_data(shape=[None,224,224,3], name='input',data_preprocessing=img_prep)

