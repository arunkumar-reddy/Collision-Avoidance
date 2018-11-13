#!/usr/bin/env python
import rospy;
from sensor_msgs.msg import LaserScan;

def sensor_callback(data):
    sensor_data = list(data.ranges);
    print(len(sensor_data));
    '''
    sensor_data = sensor_data[90:450];
    lidar_data = list();
    for i in range(0,len(sensor_data),2):
        lidar_data.append((sensor_data[i]+sensor_data[i+1])/2);
    print(len(lidar_data));
    '''
    
rospy.init_node('controller',anonymous=False);
sensor = rospy.Subscriber('/r1/sim_lms2xx_1_laserscan', LaserScan,sensor_callback);
rate = rospy.Rate(10);
rospy.spin();



    
