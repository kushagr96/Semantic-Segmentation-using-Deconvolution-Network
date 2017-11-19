import os
import random
import tensorflow as tf
import time
import numpy as np
from random import choice
from sklearn.model_selection import KFold, cross_val_score

class Segment(object):

    path = ""

    def __init__(self, data_dir,checkpoint_dir='./checkpoints/'):
        self.graph()
        self.saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allocator_type = 'BFC'
        self.session = tf.Session(config = config)
        self.session.run(tf.initialize_all_variables())
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir

    def get_image(self, image, mask_image):	
        image_placeholder = tf.placeholder(tf.string)
        mask_image_placeholder = tf.placeholder(tf.string)
        feed_dict_to_use = {image_placeholder: image, mask_image_placeholder: mask_image}
        image_tensor = tf.read_file(image_placeholder)
        mask_tensor = tf.read_file(mask_image_placeholder)
        image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
        mask_tensor = tf.image.decode_jpeg(mask_tensor, channels=1)
        mask_class = tf.to_float(tf.equal(mask_tensor, 1))
        mask_background = tf.to_float(tf.not_equal(mask_tensor, 1))
        mask = tf.concat(concat_dim=2, values=[mask_class, mask_background])
        flat_labels = tf.reshape(tensor=mask, shape=(-1, 2))
        return [image_tensor,mask_tensor,flat_labels,feed_dict_to_use]

    def iou(self,prediction, actual):
        union = np.sum(np.bitwise_or(prediction, actual))
        intersection = np.sum( np.bitwise_and(prediction,actual))
        return float(intersection)/union

    def train(self, train_indices, training_steps=50):
        """
            Trains the model on data

        """
        step_start = 0
        for i in range(step_start, step_start+training_steps):
            # pick random line from file
            j = choice(train_indices)
            image = self.data_dir + 'train-' + str(j) + '.jpg'
            mask_image = self.data_dir + 'train-' + str(j) + '-mask.jpg'
            [image_tensor, mask_tensor, flat_labels, feed_dict_to_use]=self.get_image(image, mask_image)
            train_image, train_mask, flat_labels = self.session.run([image_tensor, mask_tensor, flat_labels],feed_dict=feed_dict_to_use)
            print('run train step: '+str(i))
            start = time.time()
            feed_dict_to_use = {self.x: [train_image], self.flat_labels:[flat_labels]}
            self.train_step.run(session=self.session, feed_dict=feed_dict_to_use)
            print('step {} finished in {:.2f} s with loss of {:.6f} '.format(i, time.time() - start, self.loss.eval(session=self.session, feed_dict=feed_dict_to_use)) )
            self.save_model(i)
            actual = tf.argmax(flat_labels,1)
            print( 'IOU : {:.6f} '.format( self.iou(self.prediction.eval(session=self.session, feed_dict=feed_dict_to_use), actual.eval(session = self.session))))
            print('Model {} saved'.format(i))	



    def save_model(self, step):
        """
            saves model on the disk
        """

    	self.saver.save(self.session, self.checkpoint_dir+'model', global_step=step)


    
    def load_model(self):
        """
            returns a pre-trained instance of Segment class
        """
    	path = tf.train.get_checkpoint_state(self.checkpoint_dir)
    	self.saver.restore(self.session, path.model_checkpoint_path)
        global_step = int(path.model_checkpoint_path.split('-')[-1])
        return global_step



    def predict(self, image):
        self.load_model()
        return self.prediction.eval(session=self.session, feed_dict={self.x: [image]})

    def test(self, test_indices):
	count = 0
	for j in test_indices:
	    image = self.data_dir + 'valid-' + str(j) + '.jpg'
            mask_image = self.data_dir + 'valid-' + str(j) + '-mask.jpg'
            [image_tensor, mask_tensor, flat_labels, feed_dict_to_use]=self.get_image(image, mask_image)
            test_image, test_mask, flat_labels = self.session.run([image_tensor, mask_tensor, flat_labels],feed_dict=feed_dict_to_use)
	    actual = tf.argmax(flat_labels, 1)
	    predicted = self.predict(test_image)
	    iou = self.iou(predicted ,actual.eval(session = self.session))
	    print( iou)
	    if iou > 0.5:
       		count += 1
	return count/len(test_indices)

    def graph(self):

    	with tf.device('/gpu:1'):

            	self.x = tf.placeholder(tf.float32, shape=(1,None, None, 3))
	    	self.flat_labels = tf.placeholder(tf.int64, shape=(1,None,2))
	    	labels = tf.reshape(self.flat_labels, (-1, 2))

	    	#conv layer 1
	    	W_shape = [2, 2, 3, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_1 =  tf.nn.relu(tf.nn.conv2d(self.x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		#conv layer 2
	    	W_shape = [3, 3, 64, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_2 =  tf.nn.relu(tf.nn.conv2d(conv_1, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		# pool layer 1
		pool_1, pool_1_argmax = tf.nn.max_pool_with_argmax(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


		#conv layer 3
	    	W_shape = [3, 3, 64, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_3 =  tf.nn.relu(tf.nn.conv2d(pool_1, W, strides=[1, 1, 1, 1], padding='SAME') + b)

	    	#conv layer 4
	    	W_shape = [3, 3, 128, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_4 =  tf.nn.relu(tf.nn.conv2d(conv_3, W, strides=[1, 1, 1, 1], padding='SAME') + b)


		# pool layer 2
		pool_2, pool_2_argmax = tf.nn.max_pool_with_argmax(conv_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	    
	    	#conv layer 5
	    	W_shape = [3, 3, 128, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_5 =  tf.nn.relu(tf.nn.conv2d(pool_2, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		#conv layer 6
	    	W_shape = [3, 3, 256, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
		b = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]))
		conv_6 =  tf.nn.relu(tf.nn.conv2d(conv_5, W, strides=[1, 1, 1, 1], padding='SAME') + b)

		pool_3, pool_3_argmax = tf.nn.max_pool_with_argmax(conv_6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		dropout = tf.nn.dropout(pool_3, keep_prob = 0.5)

		#unpool 3
		padding = 'SAME'
		unpool_3 = self.unpool(dropout, pool_3_argmax, tf.shape(conv_6))


		    #deconv layer 6
	    	W_shape = [3, 3, 256, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = unpool_3
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
	        deconv_6 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


            	#deconv layer 5
	    	W_shape = [3, 3, 128, 256]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_6
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_5 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	unpool_2 = self.unpool(deconv_5, pool_2_argmax, tf.shape(conv_4))

            	#deconv layer 4
	    	W_shape = [3, 3, 128, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = unpool_2
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_4 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


            	#deconv layer 3
	    	W_shape = [3, 3, 64, 128]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_4
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_3 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	unpool_1 = self.unpool(deconv_3, pool_1_argmax, tf.shape(conv_2))

            	#deconv layer 2
            	W_shape = [3, 3, 64, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = unpool_1
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_2 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	#deconv layer 1
            	W_shape = [3, 3, 32, 64]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_2
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	deconv_1 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b


            	#deconv layer 0
            	W_shape = [1, 1, 2, 32]
	    	W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
	        b = tf.Variable(tf.constant(0.1, shape=[W_shape[2]]))
	        x = deconv_1
	        x_shape = tf.shape(x)
        	out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
        	score_1 = tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

            	flat_logits = tf.reshape(score_1, (-1, 2))
            	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = flat_logits, labels = labels, name='x_entropy')
            	self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
         
            	self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

            	self.prediction = tf.argmax(flat_logits,1)

   
    def unpool(self, x, argmax, out_shape):
	shape = tf.to_int64(out_shape)
        argmax = tf.pack([argmax // (shape[2] * shape[3]), argmax % (shape[2] * shape[3]) // shape[3]])
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])
        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]
        indices1 = tf.squeeze(argmax)
        indices1 = tf.transpose(tf.pack((indices1[0], indices1[1]), axis=0), perm=[3, 1, 2, 0])
        indices2 = tf.tile(tf.to_int64(tf.range(channels)), [((width + 1) // 2) * ((height + 1) // 2)])
        indices2 = tf.transpose(tf.reshape(indices2, [-1, channels]), perm=[1, 0])
        indices2 = tf.reshape(indices2, [channels, (height + 1) // 2, (width + 1) // 2, 1])
        indices = tf.reshape(tf.concat(3,  [indices1, indices2]), [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])
        inter = tf.reshape(tf.squeeze(x), [-1, channels])
        values = tf.reshape(tf.transpose(inter, perm=[1, 0]), [-1])
        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)
   

if __name__ == "__main__":

        obj = Segment('./Train_Data/')
	# training the model
	X = range(0,164)
        k_fold = KFold(n_splits=5)
	i = 0
	for train_indices, test_indices in k_fold.split(X):
		print( '{} validation'.format(i))
		i += 1
    		print('Train: %s | test: %s' % (train_indices, test_indices))
		obj.train(train_indices)
		print 'Accuracy = {:.3f}%'.format(obj.test(test_indices) * 100)
       	
       	# testing the model
	T = range(0,5)
	print( 'Accuracy = {:.3f}%'.format(obj.test(T) * 100))
        
