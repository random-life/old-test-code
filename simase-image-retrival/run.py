from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tflearn
import inference
import os
import h5py

hf = h5py.File('sketches-train.h5')
sketch_x = hf['X']
sketch_y = hf['Y']
sketch_x, sketch_y = tflearn.data_utils.shuffle(sketch_x, sketch_y)
hf2 = h5py.File('edges-train.h5')
edges_x = hf2['X']
edges_y = hf2['Y']
edges_x, edges_y = tflearn.data_utils.shuffle(edges_x, edges_y)

sess = tf.InteractiveSession()

siamese = inference.siamese()


train_op = tf.train.AdamOptimizer(0.0001).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

batch_size = 32
epoch = 10

steps_per_epoch= edges_x.shape[0]//batch_size
sum_steps = steps_per_epoch * epoch
print("sum_steps: ",sum_steps)

for i in range(sum_steps):
	step = i % steps_per_epoch
	step2 = i% (sketch_x.shape[0]//batch_size)
	
	idx = np.random.permutation(164)
	batch_sketch_x,batch_sketch_y = sketch_x[idx[0:32],:,:,:], sketch_y[idx[0:32],:]
	idx2 = np.random.permutation(edges_x.shape[0])
	edges_x,edges_y = edges_x[idx2,:,:,:], edges_y[idx2,:]
	#batch_sketch_x, batch_sketch_y=sketch_x[step2 * batch_size:(step2+1) * batch_size], sketch_y[step2 * batch_size:(step2+1) * batch_size]
	#batch_edges_x, batch_edges_y = edges_x[step * batch_size:(step+1) * batch_size], edges_y[step * batch_size:(step+1) * batch_size]
	#print(np.argmax(batch_sketch_y,1))
	batch_edges_x = []
	batch_y = []
	count = []
	j = 0
	sum_equal = 0
	sum_noequal = 0
	l = 0
	while(1):
		l = l % edges_x.shape[0]
		if (np.argmax(batch_sketch_y[j]) != np.argmax(edges_y[l])) and sum_noequal < 16:
				batch_y.append(0.)
				batch_edges_x.append(edges_x[l])
				j += 1
				sum_noequal += 1
				count.append(l)
		elif (np.argmax(batch_sketch_y[j]) == np.argmax(edges_y[l])) and sum_equal < 16:
				batch_y.append(1.)
				batch_edges_x.append(edges_x[l])
				sum_equal += 1
				j += 1
				count.append(l)
		l += 1
		
		if j >= 32: 
			break
	batch_edges_x = np.array(batch_edges_x)
	batch_y = np.array(batch_y)
	#print(batch_edges_x.shape,batch_y.sum(),count,j,l,edges_x.shape[0])
	#print(np.argmax(batch_edges_y,1))
	#batch_y = (np.argmax(batch_sketch_y,1)==np.argmax(batch_edges_y,1))
	#print(batch_y)
	batch_y = batch_y.astype('float')
	_, loss_v = sess.run([train_op, siamese.loss], feed_dict={
                        siamese.x1: batch_sketch_x,
                        siamese.x2: batch_edges_x,
			siamese.y_: batch_y})
	if np.isnan(loss_v):
        	print('Model diverged with loss = NaN')
        	quit()

	if step % 10 == 0:
		print ('step %d: loss %.3f' % (step, loss_v))
	'''if ((i+1) % steps_per_epoch) == 0:

		edges_x, edges_y = tflearn.data_utils.shuffle(edges_x, edges_y)
	if((step2+1) ==10):
		sketch_x, sketch_y = tflearn.data_utils.shuffle(sketch_x, sketch_y)'''

saver.save(sess, './model3')		
