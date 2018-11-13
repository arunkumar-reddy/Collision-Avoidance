#!/usr/bin/env python
import sys;
import rospy;
import random;
import copy;
import time;
import math;
import numpy as np;
from geometry_msgs.msg import Twist;
from geometry_msgs.msg import Pose;
from sensor_msgs.msg import LaserScan;
from nav_msgs.msg import Odometry;
from gazebo_msgs.msg import ContactsState;
from cadrl.msg import *;
from cadrl.srv import *;
from tf.transformations import euler_from_quaternion;

class Robot:
	def __init__(self,name):
		rospy.init_node(name,anonymous=False);
		self.name = name;
		self.map = 'robita1';
		self.pose = Pose();
		self.pose.position.x = 100;
		self.pose.position.y = 100;
		self.velocity = Twist();
		self.laser_scan = [0.0]*180;
		self.laser_derivative = [0.0]*180;
		self.collision = False;
		self.obstacles = [Pose() for i in range(0,9)];
		self.controller = rospy.Publisher('/'+name+'/cmd_vel',Twist,queue_size=10);
		self.sensor = rospy.Subscriber('/'+name+'/laser/scan', LaserScan, self.laser_callback);
		self.contact = rospy.Subscriber('/'+name+'/laser/bumper', ContactsState, self.collision_callback);
		self.odometer = rospy.Subscriber('/'+name+'/odom', Odometry, self.position_callback);
		self.detector = rospy.Subscriber('/dynamic',Obstacles, self.obstacle_callback);
		rospy.wait_for_service('run');
		rospy.wait_for_service('train');
		rospy.wait_for_service('estimate');
		self.run = rospy.ServiceProxy('run',NeuralNetworkRun);
		self.estimate = rospy.ServiceProxy('estimate',NeuralNetworkRun);
		self.train = rospy.ServiceProxy('train',NeuralNetworkTrain);
		self.rate = rospy.Rate(10);
		self.state_dim = 360;
		self.num_action = 5;
		self.angular = 1.2;
		if(self.map=='robita'):
			self.goals = [(0,0),(14,7.5),(11,-12.5),(-12,2)];
		elif(self.map=='robita1'):
			self.goals = [(0,0),(9,5),(7,-8),(-7,2)];
		else:
			self.goals = [(2,0),(7,-4),(-4,0),(0,-4),(-11,0),(-6.5,-4),(-2,2),(-4,4),(6.7,4.0),(-11,4.0)];
		self.index = -1;
		self.goal = Pose();
		if(self.name=='r8'):
			self.pref_speed = 0.5;
		elif(self.name=='r9'):
			self.pref_speed = 1.0;
		else:
			self.pref_speed = random.uniform(0.5,1.0);
		self.linear = self.pref_speed;
		self.new_goal();
		print('Created Robot '+str(name));

	def laser_callback(self,data):
		lidar_data = list(data.ranges);
		for i in range(0,len(lidar_data)):
			if(math.isinf(lidar_data[i]) or math.isnan(lidar_data[i])):
				lidar_data[i] = 5.0;
			elif(lidar_data[i]<0.0):
				lidar_data[i] = 0.0;
			elif(lidar_data[i]>5.0):
				lidar_data[i] = 5.0;
		self.laser_derivative = [(x-y) for x,y in zip(lidar_data,self.laser_scan)];
		self.laser_scan = lidar_data;

	def position_callback(self,data):
		self.pose = data.pose.pose;

	def obstacle_callback(self,data):
		self.obstacles = data.positions;

	def collision_callback(self,data):
		if(len(data.states)>0):
			self.collision = True;
		else:
			flag = False;
			for i in range(0,len(self.laser_scan)):
				if(self.laser_scan[i]<=0.1):
					flag = True;
					break;
			if(flag==True):
				self.collision = True;
			else:
				self.collision = False;

	def distance(self,x1,y1,x2,y2):
		return math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2));

	def reached(self):
		tolerance = 1.0;
		if(self.distance(self.pose.position.x,self.pose.position.y,self.goal.position.x,self.goal.position.y)<tolerance):
			return True;
		return False;

	def difference(self,angle,theta):
		if(abs(angle-theta)<=3.14):
			return angle-theta;
		elif((angle-theta)>3.14):
			return -(6.28-(angle-theta));
		else:
			return 6.28+(angle-theta);

	def a2g(self):
		x = self.pose.position.x;
		y = self.pose.position.y;
		quaternion = (self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w);
		euler = euler_from_quaternion(quaternion);
		theta = euler[2];
		force = [0.0,0.0];
		for i in range(0,len(self.laser_scan)):
			dist = self.laser_scan[i];
			angle = (i-90)*(3.14/180);
			force[0] -= (dist*math.cos(theta+angle))/pow(dist+0.1,2);
			force[1] -= (dist*math.sin(theta+angle))/pow(dist+0.1,2);
		for i in range(0,len(self.obstacles)):
			dist = self.distance(x,y,self.obstacles[i].position.x,self.obstacles[i].position.y);
			if(dist<=1.0):
				euler = euler_from_quaternion((self.obstacles[i].orientation.x,self.obstacles[i].orientation.y,self.obstacles[i].orientation.z,self.obstacles[i].orientation.w));
				diff1 = self.difference(theta,euler[2]);
				diff2 = self.difference(theta,np.arctan2(self.obstacles[i].position.y-y,self.obstacles[i].position.x-x));
				if(abs(diff1)>2.6 and abs(diff2)<0.4):
					force[0] += 20.0*dist*(math.cos(theta+1.04))/pow(dist+0.1,2);
					force[1] += 20.0*dist+(math.sin(theta+1.04))/pow(dist+0.1,2);
				elif(abs(diff1)<0.4 and abs(diff2)<0.4):
					force[0] += 20.0*dist*(math.cos(theta+1.04))/pow(dist+0.1,2);
					force[1] += 20.0*dist*(math.cos(theta+1.04))/pow(dist+0.1,2);
				elif(abs(diff1)<0.4 and abs(diff2)>2.6):
					force[0] += 20.0*dist*(math.cos(theta-1.04))/pow(dist+0.1,2);
					force[1] += 20.0*dist*(math.cos(theta-1.04))/pow(dist+0.1,2);
		dist = self.distance(x,y,self.goal.position.x,self.goal.position.y);
		force[0] += 50.0*(self.goal.position.x-x)/(dist+0.01);
		force[1] += 50.0*(self.goal.position.y-y)/(dist+0.01);
		angle = np.arctan2(force[1],force[0]);
		return self.difference(angle,theta);

	def new_goal(self):
		if(self.map=='robita' or self.map=='robita1'):
			if(self.index==-1):
				self.index = 0;
			elif(self.index==0):
				eps = random.random();
				self.index = 1 if eps<=0.33 else 2 if eps<=0.66 else 3;
			elif(self.index==1):
				self.index = 0;
			elif(self.index==2):
				self.index = 0;
			elif(self.index==3):
				self.index = 0;
			self.goal.position.x = self.goals[self.index][0];
			self.goal.position.y = self.goals[self.index][1];
		else:
			if(self.index==-1):
				x = self.pose.position.x;
				y = self.pose.position.y;
				if(1<=x<=8 and -4.5<=y<=-1):
					self.index = 0;
				elif(-5<=x<=1 and -4.5<=y<=-1):
					self.index = 2;
				elif(-6<=x<=-12 and -4.5<=y<=-1):
					self.index = 4;
				elif(-5<=x<=1 and 2<=y<=5):
					self.index = 6;
				elif(1<=x<=8 and 0<=y<=5):
					self.index = 0;
				else:
					self.index = 9;
			elif(self.index==0):
				eps = random.random();
				self.index = 1 if eps<0.3 else 8 if eps<0.6 else 6;
			elif(self.index==1):
				self.index = 0;
			elif(self.index==2):
				eps = random.random();
				self.index = 3 if eps<0.3 else 0 if eps<0.6 else 6;
			elif(self.index==3):
				self.index = 2;
			elif(self.index==4):
				eps = random.random();
				self.index = 5 if eps<0.3 else 0 if eps<0.6 else 6;
			elif(self.index==5):
				self.index = 4;
			elif(self.index==6):
				eps = random.random();
				self.index = 7 if eps<0.3 else 0 if eps<0.6 else 2;
			elif(self.index==7):
				self.index = 6;
			elif(self.index==8):
				self.index = 0;
			elif(self.index==9):
				eps = random.random();
				self.index = 2 if eps<0.5 else 4;
			self.goal.position.x = self.goals[self.index][0];
			self.goal.position.y = self.goals[self.index][1];

	def dynamic(self):
		tolerance = 0.6;
		reward = 0.0;
		x = self.pose.position.x;
		y = self.pose.position.y;
		euler = euler_from_quaternion((self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w));
		theta = euler[2];
		idx = int(self.name[-1]);
		for i in range(0,len(self.obstacles)):
			if((i+1)==idx):
				continue;
			dist = self.distance(x,y,self.obstacles[i].position.x,self.obstacles[i].position.y);
			if(dist<=tolerance):
				reward += -0.1*(0.3/max(0.3,dist));
			'''
			elif(dist<=1.0):
				euler = euler_from_quaternion((self.obstacles[i].orientation.x,self.obstacles[i].orientation.y,self.obstacles[i].orientation.z,self.obstacles[i].orientation.w));
				angle = euler[2];
				diff = self.difference(theta,angle);
				k1 = self.obstacles[i].position.x-x;
				k2 = self.obstacles[i].position.y-y;
				x1 = k1*math.cos(diff)+k2*math.sin(diff);
				y1 = -k1*math.sin(diff)+k2*math.cos(diff);
				if(-0.3<=x1<=0 and 0<=y1<=0.5 and abs(diff)>2.5):
					reward += -0.1;
				elif(0<=x1<=0.3 and 0<=y1<=0.5 and abs(diff)<0.5):
					reward += -0.1;
			'''
		return reward;
		
	def get_reward(self):
		angle = self.a2g();
		if self.reached():
			print(self.name+' Reached goal');
			self.new_goal();
			time.sleep(0.3);
			return 1;
		elif self.collision:
			return -1;
		reward = self.dynamic();
		angle = self.a2g();
		dist = self.distance(self.pose.position.x,self.pose.position.y,self.goal.position.x,self.goal.position.y);
		reward += -1.0*abs(angle/20);
		return max(-1,reward);

	def action(self,choice,angle=None):
		if(choice==0):
			self.velocity.linear.x = self.linear;
			self.velocity.angular.z = 0;
		elif(choice==1):
			self.velocity.linear.x = self.linear/2;
			self.velocity.angular.z = -self.angular;
		elif(choice==2):
			self.velocity.linear.x = self.linear/2;
			self.velocity.angular.z = self.angular;
		elif(choice==3):
			self.linear = min(self.pref_speed,self.linear+0.1);
			self.velocity.linear.x = self.linear;
			self.velocity.angular.z = 0;
		elif(choice==4):
			self.linear = max(0.3,self.linear-0.1);
			self.velocity.linear.x = self.linear;
			self.velocity.angular.z = self.angular if angle>0 else -self.angular;
		elif(choice==5):
			self.velocity.linear.x = -10.0;
			self.velocity.angular.z = 0;
			for i in range(0,15):
				self.controller.publish(self.velocity);
				time.sleep(0.05);
			return;
		elif(choice==6):
			self.velocity.linear.x = 0;
			self.velocity.angular.z = 3.0 if random.random()>0.5 else -5.0;
			self.controller.publish(self.velocity);
			time.sleep(0.05);
			return;
		elif(choice==7):
			self.velocity.linear.x = 0;
			self.velocity.angular.z = 0;
			self.controller.publish(self.velocity);
			time.sleep(0.05);
			return;
		self.controller.publish(self.velocity);
		time.sleep(0.05);
		return self.get_reward();

	def apf(self):
		angle = self.a2g();
		if(abs(angle)<0.35):
			if(self.linear<self.pref_speed):
				return 3,self.action(3);
			else:
				return 0,self.action(0);
		elif(abs(angle)>2.0):
			return 4,self.action(4,angle);
		elif(angle>0):
			return 2,self.action(2);
		else:
			return 1,self.action(1);

	def inputs(self):
		scan = self.laser_scan;
		derivative = self.laser_derivative;
		scan = (scan-np.mean(scan))/(np.std(scan)+0.01);
		derivative = (derivative-np.mean(derivative))/(np.std(derivative)+0.01);
		state = np.concatenate((scan,derivative));
		angle = self.a2g();
		return list(state),angle;

	def qlearn(self):
		episodes = 0;
		action_steps = 1000;
		discount = 0.8;
		states = np.vstack([np.array([5*random.random() for i in range(self.state_dim)]).reshape(1,-1) for i in range(action_steps)]);
		angles = np.vstack([np.array([random.random() for i in range(1)]).reshape(1,-1) for i in range(action_steps)]);
		targets = np.vstack([np.array([1 for i in range(self.num_action)]).reshape(1,-1) for i in range(action_steps)]);
		while not rospy.is_shutdown():
			longterm_reward = 0.0;
			for i in range(0,action_steps):
				curr_state,angle = self.inputs();
				output = self.run(NeuralNetworkInput(curr_state,angle));
				values = output.actions.actions;
				epsilon = random.random();
				if(epsilon<0.6):
					action,reward = self.apf();
				elif(epsilon<1.0):
					action = random.choice([0,1,2,3,4]);
					reward = self.action(action);
				else:
					action = np.argmax(values);
					reward = self.action(action);
				if(reward==-1):
					self.action(5);
					self.action(6);
				longterm_reward += reward;
				next_state,next_angle = self.inputs();
				target = np.array(values);
				estimate = self.estimate(NeuralNetworkInput(next_state,next_angle));
				if(reward==-1 or reward==1):
					target[action] = reward;
				else:
					target[action] = reward+discount*(np.max(estimate.actions.actions));
				states = np.vstack((states,curr_state));
				angles = np.vstack((angles,angle));
				targets = np.vstack((targets,target));
				if((i+1)%200==0):
					print('Q-values:'+str(values)+' Reward:'+str(reward)); 

			self.action(7);
			states = states[action_steps:];
			angles = angles[action_steps:];
			targets = targets[action_steps:];
			batch = NeuralNetworkBatch(list(states.flatten()),list(angles.flatten()),list(targets.flatten()));
			loss = self.train(batch);
			print('Episode: '+str(episodes+1)+' Loss: '+str(loss));
			episodes += 1;

	def test(self):
		while not rospy.is_shutdown():
			if(self.collision==True):
				self.action(5);
				self.action(6);
			curr_state,angle = self.inputs();
			output = self.run(NeuralNetworkInput(curr_state,angle));
			values = output.actions.actions;
			action = np.argmax(values);
			reward = self.action(action);
			#print('Q-Values: '+str(values));
		
name = sys.argv[1];
train = False;
robot = Robot(name);
if(train==True):
	robot.qlearn();
else:
	robot.test();
