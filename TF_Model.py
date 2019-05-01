import tensorflow as tf
import numpy as np 
import sys
import time
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix


class CNN_Model:

    def __init__(self, xtr,ytr,xte,yte):

        # Constants for the Model
        self.n_classes = 43
        self.filter_x = 5
        self.filter_y = 5
        self.image_h = 32
        self.image_w = 32
        self.feature_map_in = 1
        self.feature_map_1 = 50
        self.feature_map_2 = 100
        self.feature_map_3 = 150
        self.feature_map_4 = 200
        self.fc_ = 1024
        self.stride_conv = 1
        self.stride_mpool = 2
        self.init = tf.global_variables_initializer()
        self.epochs_ = 50
        self.batch_ = 256

        # Variables from outside
        self.xtr = xtr
        self.ytr = ytr
        self.xte = xte
        self.yte = yte

        # Model Weights and Biases
        self.weights = {
                    'con_1' : tf.Variable(tf.random_normal([self.filter_x,self.filter_y,self.feature_map_in,self.feature_map_1])),
                    'con_2' : tf.Variable(tf.random_normal([self.filter_x,self.filter_y,self.feature_map_1,self.feature_map_2])),
                    'con_3' : tf.Variable(tf.random_normal([self.filter_x,self.filter_y,self.feature_map_2,self.feature_map_3])),
                    'con_4' : tf.Variable(tf.random_normal([self.filter_x,self.filter_y,self.feature_map_3,self.feature_map_4])),
                    'fc' : tf.Variable(tf.random_normal([np.int(self.image_h/16) * np.int(self.image_w/16) * self.feature_map_4, self.fc_])),
                    'output' : tf.Variable(tf.random_normal([self.fc_,self.n_classes]))}

        self.biases = {
                    'con_1' : tf.Variable(tf.random_normal([self.feature_map_1])),
                    'con_2' : tf.Variable(tf.random_normal([self.feature_map_2])),
                    'con_3' : tf.Variable(tf.random_normal([self.feature_map_3])),
                    'con_4' : tf.Variable(tf.random_normal([self.feature_map_4])),
                    'fc' : tf.Variable(tf.random_normal([self.fc_])),
                    'output' : tf.Variable(tf.random_normal([self.n_classes]))}            
        return None


    def minibatch(self,x,y, b_size, shuffle = True):
        if x.shape[0] != y.shape[0]:
            print("Predictor - Response Mismatch, Program Terminating...")
            sys.exit()
        else:
            n_samples = x.shape[0]
            if shuffle:
                idx = np.random.permutation(n_samples)
            else:
                idx = list(range(n_samples))
            for k in range(int(np.ceil(n_samples/b_size))):
                from_idx=k*b_size
                to_idx=(k+1)*b_size
                yield x[idx[from_idx:to_idx],:,:,:], y[idx[from_idx:to_idx],:]


    # Convolution 2D
    def conv_2d(self,input_tensor,kernel,bias):
        tensor = tf.nn.conv2d(input_tensor,kernel,strides=[1,self.stride_conv,self.stride_conv,1],padding='SAME')
        tensor = tf.nn.bias_add(tensor,bias)
        tensor = tf.nn.leaky_relu(tensor)
        return tensor


    # Max Pooling
    def maxpool_2d(self,input_tensor):
        tensor = tf.nn.max_pool(input_tensor,ksize = [1,self.stride_mpool,self.stride_mpool,1], strides = [1,self.stride_mpool,self.stride_mpool,1], padding = 'SAME')
        return tensor


    # Fully Connected Layer
    def fc_layer(self,input_tensor,kernel,bias,activation):
        tensor = tf.add(tf.matmul(input_tensor,kernel),bias)
        if activation:
            tensor = tf.nn.leaky_relu(tensor)
        return tensor


    def Neural_Net(self,image,kernel_dict,bias_dict):
        print("Input shape :", image.get_shape())
        
        # Layer 1
        conv_1 = self.conv_2d(image,kernel_dict['con_1'],bias_dict['con_1'])
        print("Conv1 Shape: ",conv_1.get_shape())
        max_1 = self.maxpool_2d(conv_1)
        print("Max1 Shape: ",max_1.get_shape())
        
        # Layer 2
        conv_2 = self.conv_2d(max_1,kernel_dict['con_2'],bias_dict['con_2'])
        print("Conv2 Shape: ",conv_2.get_shape())
        max_2 = self.maxpool_2d(conv_2)
        print("Max2 Shape: ",max_2.get_shape())
        
        # Layer 3
        conv_3 = self.conv_2d(max_2,kernel_dict['con_3'],bias_dict['con_3'])
        print("Conv3 Shape: ",conv_3.get_shape())
        max_3 = self.maxpool_2d(conv_3)
        print("Max3 Shape: ",max_3.get_shape())
        
        # Layer 4
        conv_4 = self.conv_2d(max_3,kernel_dict['con_4'],bias_dict['con_4'])
        print("Conv4 Shape: ",conv_4.get_shape())
        max_4 = self.maxpool_2d(conv_4)
        print("Max4 Shape: ",max_4.get_shape())
        
        # Layer FC
        fc = tf.reshape(max_4,[-1,kernel_dict['fc'].get_shape().as_list()[0]])
        fc = self.fc_layer(fc,kernel_dict['fc'],bias_dict['fc'], False)
        print("fc Shape: ",fc.get_shape())
        
        # Output Layer
        output = tf.add(tf.matmul(fc,kernel_dict['output']),bias_dict['output'])
        print("output Shape: ",output.get_shape())
        return output

    def Training(self):
        # Defining Tensorflow Placeholders
        x = tf.placeholder(tf.float32,shape = (None,self.image_h,self.image_w,1))
        y = tf.placeholder(tf.float32,shape = (None,self.n_classes))
        logits = self.Neural_Net(x,self.weights,self.biases)
        pred_ = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # Launching Graph
        start_ = time.time()
        with tf.Session() as sess:
            sess.run(self.init)
            for i in tqdm(range(self.epochs_)):
                for mb in self.minibatch(self.xtr,self.ytr,self.batch_, True):
                    tf_output = sess.run([optimizer,loss], feed_dict = {x:mb[0],y:mb[1]})
            print("Calibration Done")

            y_hat = sess.run(pred_, feed_dict = {x:self.xte})

        end_ = time.time()
        total_time = np.round((end_-start_),0)
        print("Total Time Taken: ",total_time//60," Minutes and ",total_time%60," Seconds")
        return y_hat,self.yte
