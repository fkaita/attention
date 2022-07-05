#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import rospy
import math
import sys
import dlib
from imutils import face_utils
import time
import moveit_commander
import geometry_msgs.msg
from std_msgs.msg import String
import rosnode
import actionlib
from hand_gripper import *
from neck_yaw_pitch import *
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState)
from trajectory_msgs.msg import JointTrajectoryPoint


def main():
    rospy.init_node("sciurus17_pick_and_place_controller")
    handgripper = HandGripper()
    led_pub = rospy.Publisher('neopixel_simple', String , queue_size=1)
    neck = NeckYawPitch()
    
    while not neck.state_received():
        pass
    yaw_angle = neck.get_current_yaw()
    pitch_angle = neck.get_current_pitch()
    
    # Initialize neck
    neck.set_angle(math.radians(0), math.radians(0),goal_secs=1.0)
       
#    command = String("br")
#    led_pub.publish(command)
#    time.sleep(3)
#    
#    command = String("s")
#    led_pub.publish(command)
    
    # Initialize pose and gripper open->close
    handgripper.init_pose()
    handgripper.set_gripper(goal=0.8, right_left=1)
    handgripper.set_gripper(goal=0.0, right_left=1)
    handgripper.set_gripper(goal=0.8, right_left=0)
    handgripper.set_gripper(goal=0.0, right_left=0)
    
    targets = [[0.3,0.07,0.19], [0.30,-0.08,0.19],[0.44,0.07,0.2], [0.2, -0.02, 0.18], [0.39, -0.02, 0.2]]
    goals = [[0.33,0.29,0.24], [0.40,0.29,0.23], [0.48,0.29,0.25], [0.34,0.37,0.25],  [0.47,0.36,0.25]]
    i = 0
    for target in targets:
        goal = goals[i]
    
        # Look at the object
        yaw_angle = 0
        pitch_angle = -30
        neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle),goal_secs=1.0)

        # Set hand above object
        handgripper.set_hand([target[0],target[1],0.4], [3.14/2.0, 0.0, 0.0], right_left=1)
        
        # Open gripper
        handgripper.set_gripper(goal=0.85, right_left=1)
        time.sleep(1)
        
        # Reach hand to object
        handgripper.set_hand(target, [3.14/2.0, 0.0, 0.0], right_left=1)
        
        # Close gripper
        handgripper.set_gripper(goal=0.42, right_left=1)
        
        # Check if object in hand or not
        right_left = 1
        hand_currents = []
        for j in range(10):
	        current = handgripper.get_hand_current()[right_left]
	        hand_currents.append(current)
	            
        mean_current = np.mean(hand_currents)
        print('mean current is ', mean_current)
        
        if i==2 or i==4:
            # Fail 
#            handgripper.set_hand([target[0],target[1],0.4], [3.14/2.0, 0.0, 0.0], right_left=1)
            handgripper.set_hand([0.52,0.17,0.5], [3.14/2.0, 0.0, 0.0], right_left=1)
            handgripper.set_gripper(goal=0.52, right_left=1)
            
            # Set hand over place to put
            handgripper.set_hand([0.36, 0.33, 0.4], [3.14/2.0, 0.0, 0.0], right_left=1)
            
            # Look at the place to put
            yaw_angle = 15
            pitch_angle = -30 
            neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle),goal_secs=1.0)
            time.sleep(4)
            i += 1
            
            for angle in [[-10,-30], [10,-30],[0,10]]:
                # angle (yaw, pitch)
                neck.set_angle(math.radians(angle[0]), math.radians(angle[1]),goal_secs=1.0)
                time.sleep(4)
            
            handgripper.set_hand([0.5,0.1,0.4], [3.14/2.0, 3.14/2.0, 0.0], right_left=1)
            
            for gripper_goal in [0.8, 0.42, 0.8, 0.42]:
                handgripper.set_gripper(goal=gripper_goal, right_left=1)
                time.sleep(4)           
            continue
        else:
            # Put hand up
            handgripper.set_hand([target[0],target[1],0.4], [3.14/2.0, 0.0, 0.0], right_left=1)
        
        # Look at the place to put
        yaw_angle = 15
        pitch_angle = -30 
        neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle),goal_secs=1.0)
       
        # Set hand over place to put
        handgripper.set_hand([0.36, 0.33, 0.4], [3.14/2.0, 0.0, 0.0], right_left=1)
        handgripper.set_hand([goal[0],goal[1],0.35], [3.14/2.0, 0.0, 0.0], right_left=1)

        # Put hand down to targeted place 
        handgripper.set_hand(goal, [3.14/2.0, 0.0, 0.0], right_left=1)
        
        # Open hand
        handgripper.set_gripper(goal=0.8, right_left=1)
        time.sleep(1)   
        
        # Put hand up
        handgripper.set_hand([goal[0],goal[1],0.37], [3.14/2.0, 0.0, 0.0], right_left=1)
        handgripper.set_hand([0.36, 0.33, 0.42], [3.14/2.0, 0.0, 0.0], right_left=1)
        
        # Look in front
        neck.set_angle(math.radians(0), math.radians(0), goal_secs=1.0)
        
        i += 1
        
    # Set hand init pose
    handgripper.init_pose()
    
#    ## LED GREEN
#    command = String("bg")
#    led_pub.publish(command)
#    time.sleep(5)
#    
#    command = String("s")
#    led_pub.publish(command)
    
    print("done")


if __name__ == '__main__':
    try:
        if not rospy.is_shutdown():
            main()
    except rospy.ROSInterruptException:
        pass
