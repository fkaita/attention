#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import String
import time


rospy.init_node('led')
pub = rospy.Publisher('neopixel_simple', String , queue_size=1)

rate = rospy.Rate(10)

while not rospy.is_shutdown(): 
    command = String("br")
    pub.publish(command)
    time.sleep(5)
    
    command = String("bg")
    pub.publish(command)
    time.sleep(5)
    
    command = String("bb")
    pub.publish(command)
    time.sleep(6)
    
    rate.sleep()

if rospy.is_shutdown():     
    command = String("sc")
    pub.publish(command)
