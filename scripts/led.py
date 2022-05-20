#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import String


rospy.init_node('led')
pub = rospy.Publisher('neopixel_simple', String , queue_size=1)

rate = rospy.Rate(10)

while not rospy.is_shutdown():
    command = "bb"
    pub.publish(command)
    rate.sleep()
