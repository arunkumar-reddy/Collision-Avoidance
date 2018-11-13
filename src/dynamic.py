#!/usr/bin/env python
import rospy;
from geometry_msgs.msg import Pose;
from nav_msgs.msg import Odometry;
from cadrl.msg import Obstacles;

def callback1(data):
	positions[0] = data.pose.pose;

def callback2(data):
	positions[1] = data.pose.pose;

def callback3(data):
	positions[2] = data.pose.pose;

def callback4(data):
	positions[3] = data.pose.pose;

def callback5(data):
	positions[4] = data.pose.pose;

def callback6(data):
	positions[5] = data.pose.pose;

def callback7(data):
	positions[6] = data.pose.pose;

def callback8(data):
	positions[7] = data.pose.pose;

def callback9(data):
	positions[8] = data.pose.pose;

rospy.init_node('dynamic',anonymous=False);
positions = [Pose() for i in range(0,9)];
names = ['r'+str(i+1) for i in range(0,9)];

subscriber1 = rospy.Subscriber('/r1/odom',Odometry,callback1);
subscriber2 = rospy.Subscriber('/r2/odom',Odometry,callback2);
subscriber3 = rospy.Subscriber('/r3/odom',Odometry,callback3);
subscriber4 = rospy.Subscriber('/r4/odom',Odometry,callback4);
subscriber5 = rospy.Subscriber('/r5/odom',Odometry,callback5);
subscriber6 = rospy.Subscriber('/r6/odom',Odometry,callback6);
subscriber7 = rospy.Subscriber('/r7/odom',Odometry,callback7);
subscriber8 = rospy.Subscriber('/r8/odom',Odometry,callback8);
subscriber9 = rospy.Subscriber('/r9/odom',Odometry,callback9);
publisher = rospy.Publisher('/dynamic',Obstacles,queue_size=10);
obstacles = Obstacles();
while not rospy.is_shutdown():
	obstacles.positions = positions;
	publisher.publish(obstacles);
