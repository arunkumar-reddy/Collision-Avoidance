#!/usr/bin/env python
import rospy;
import numpy as np;
import tensorflow as tf;
from sklearn.utils import shuffle;
from cadrl.msg import *;
from cadrl.srv import *;

class CNN:
	def __init__(self):
		'''Config'''
		self.epochs = 20;
		self.batch_size = 20;
		self.state_dim = 360;
		self.num_action = 5;
		self.alpha = 0.001;
		'''Architecture'''
		self.states = tf.placeholder('float',[None,self.state_dim,1]);
		self.targets = tf.placeholder('float',[None,self.num_action]);
		self.angle = tf.placeholder('float',[None,1]);
		conv1 = tf.layers.conv1d(inputs=self.states,filters=32,kernel_size=2,strides=1,padding='same',activation=tf.nn.relu);
		conv1 = tf.layers.dropout(inputs=conv1,rate=0.1);
		pool1 = tf.layers.max_pooling1d(inputs=conv1,pool_size=2,strides=2,padding='same');
		conv2 = tf.layers.conv1d(inputs=pool1,filters=64,kernel_size=2,strides=1,padding='same',activation=tf.nn.relu);
		conv2 = tf.layers.dropout(inputs=conv2,rate=0.1);
		pool2 = tf.layers.max_pooling1d(inputs=conv2,pool_size=2,strides=2,padding='same');
		conv3 = tf.layers.conv1d(inputs=pool2,filters=64,kernel_size=2,strides=1,padding='same',activation=tf.nn.relu);
		conv3 = tf.layers.dropout(inputs=conv3,rate=0.1);
		pool3 = tf.layers.max_pooling1d(inputs=conv3,pool_size=2,strides=2,padding='same');
		conv4 = tf.layers.conv1d(inputs=pool3,filters=32,kernel_size=2,strides=1,padding='same',activation=tf.nn.relu);
		conv4 = tf.layers.dropout(inputs=conv4,rate=0.1);
		pool4 = tf.layers.max_pooling1d(inputs=conv4,pool_size=2,strides=2,padding='same');
		conv5 = tf.layers.conv1d(inputs=pool4,filters=4,kernel_size=2,strides=1,padding='same',activation=tf.nn.relu);
		conv5 = tf.layers.dropout(inputs=conv5,rate=0.1);
		pool5 = tf.layers.max_pooling1d(inputs=conv5,pool_size=2,strides=2,padding='same');
		fc = tf.reshape(pool5,[-1,48]);
		fc = tf.concat([fc,self.angle],axis=1);
		self.outputs = tf.layers.dense(fc,self.num_action);
		'''Loss'''
		self.loss = tf.reduce_mean(tf.square(self.targets-self.outputs));
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha);
		self.train_op = self.optimizer.minimize(self.loss);

	def values(self,states,angle):
		outputs = sess.run([self.outputs],feed_dict={self.states: states,self.angle: angle});
		return outputs;

	def train(self,states,angles,actions):
		loss = 0.0;
		for i in range(self.epochs):
			states,angles,actions = shuffle(states,angles,actions);
			batches = int(len(states)/self.batch_size);
			for i in range(batches):
				inputs = states[self.batch_size*i:self.batch_size*(i+1)];
				a2gs = angles[self.batch_size*i:self.batch_size*(i+1)];
				outputs = actions[self.batch_size*i:self.batch_size*(i+1)];
				cost,_ = sess.run([self.loss,self.train_op],feed_dict={self.states: inputs,self.angle: a2gs,self.targets: outputs});
				loss += cost/batches;
		return (loss/self.epochs);

def save(step):
	saver.save(sess,save_dir+'/model',global_step=step);
	print('Saved Model');

def restore():
	checkpoint = tf.train.latest_checkpoint(save_dir);
	if(checkpoint!=None):
		saver.restore(sess,checkpoint);
		print('Saved Model Restored');
		return True;
	else:
		print('No Saved Models found');
		return False;

def run(req):
	states = np.reshape(req.states.states,(1,network.state_dim,1));
	angle = np.reshape(req.states.angle,(1,1));
	outputs = np.reshape(network.values(states,angle),(network.num_action));
	res = NeuralNetworkOutput(list(outputs));
	return NeuralNetworkRunResponse(res);

def estimate(req):
	states = np.reshape(req.states.states,(1,network.state_dim,1));
	angle = np.reshape(req.states.angle,(1,1));
	outputs = np.reshape(estimator.values(states,angle),(network.num_action));
	for op in estimator_ops:
		sess.run(op);
	res = NeuralNetworkOutput(list(outputs));
	return NeuralNetworkRunResponse(res);

def train(req):
	global train_steps;
	states = np.reshape(req.states.states,(-1,network.state_dim,1));
	angles = np.reshape(req.states.angles,(-1,1));
	actions = np.reshape(req.states.actions,(-1,network.num_action));
	loss = network.train(states,angles,actions);
	train_steps += 1;
	if(train_steps%100==0):
		save(train_steps);
	return NeuralNetworkTrainResponse(loss);

rospy.init_node('cnn',anonymous=False);
network = CNN();
estimator = CNN();
estimator_ops = [];
beta = 0.01;
variables = tf.trainable_variables();
l = len(variables);
for index,var in enumerate(variables[0:l/2]):
	estimator_ops.append(variables[index+l/2].assign((var.value()*beta)+((1-beta)*variables[index+l/2].value())));

saver = tf.train.Saver();
save_dir = '/home/arun/ROS/catkin_ws/src/cadrl/models';
train_steps = 5000;
config = tf.ConfigProto();
config.gpu_options.allow_growth = True;  
sess = tf.Session(config=config);
if(restore()==False):
	sess.run(tf.global_variables_initializer());
run_network = rospy.Service('run',NeuralNetworkRun,run);
train_network = rospy.Service('train',NeuralNetworkTrain,train);
estimate_network = rospy.Service('estimate',NeuralNetworkRun,estimate);
rospy.spin();
