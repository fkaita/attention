#!/usr/bin/env python3
# coding: utf-8

import rospy
import sys
import time
import moveit_commander
import math
import geometry_msgs.msg
import rosnode
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal,
                              FollowJointTrajectoryAction, FollowJointTrajectoryGoal, 
                              JointTrajectoryControllerState)
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64



class HandGripper(object):
    def __init__(self):
        # Code for hand movement
        self.robot = moveit_commander.RobotCommander()
        self.r_arm = moveit_commander.MoveGroupCommander("r_arm_group")
        self.l_arm = moveit_commander.MoveGroupCommander("l_arm_group")
        self.arms = (self.r_arm, self.l_arm)
        for arm in self.arms:
            arm.set_max_velocity_scaling_factor(0.1)
    
        self.r_gripper = actionlib.SimpleActionClient("/sciurus17/controller1/right_hand_controller/gripper_cmd", GripperCommandAction)
        self.r_gripper.wait_for_server()
        
        if not self.r_gripper.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Gripper Action Server Not Found")
            rospy.signal_shutdown("Gripper Action Server not found")
            sys.exit(1)
                      
        self.l_gripper = actionlib.SimpleActionClient("/sciurus17/controller2/left_hand_controller/gripper_cmd", GripperCommandAction)
        self.l_gripper.wait_for_server()
            
        if not self.l_gripper.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Gripper Action Server Not Found")
            rospy.signal_shutdown("Gripper Action Server not found")
            sys.exit(1)

        self.grippers = (self.r_gripper, self.l_gripper)
    
        self.r_gripper_goal = GripperCommandGoal()
        self.r_gripper_goal.command.max_effort = 2.0
        self.l_gripper_goal = GripperCommandGoal()
        self.l_gripper_goal.command.max_effort = 2.0
        self.gripper_goals = (self.r_gripper_goal, self.l_gripper_goal)
    
        self.arm_poses = ["r_arm_init_pose", "l_arm_init_pose"]
   
        self._r_state_sub = rospy.Subscriber("/sciurus17/controller1/joints/r_hand_joint/current", 
                Float64, self._r_state_callback, queue_size=1)

        self._l_state_sub = rospy.Subscriber("/sciurus17/controller2/joints/l_hand_joint/current", 
                Float64, self._l_state_callback, queue_size=1)

        self._r_state_received = False
        self._l_state_received = False
        self._r_hand_current = 0.0
        self._l_hand_current = 0.0


    def _r_state_callback(self, state):
        self._r_state_received = True
        self._r_hand_current = state.data
#        print('self._r_hand_current is ', self._r_hand_current)

        
    def _l_state_callback(self, state):
        self._l_state_received = True
        self._l_hand_current = state.data


    def state_received(self):
        return (self._r_state_received, self._l_state_received)


    def get_hand_current(self):
        return (self._r_hand_current, self._l_hand_current)
        

    def set_hand(self, hand_pos, gripper_pos, right_left=0):
        # right_left -> 0:right hand, 1:left hand
      
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = hand_pos[0]
        target_pose.position.y = hand_pos[1] # adjustment
        target_pose.position.z = hand_pos[2]
        print('hand pos: ', hand_pos[0],hand_pos[1],hand_pos[2])
            
        arm = self.arms[right_left]
        gripper = self.grippers[right_left]
        gripper_goal = self.gripper_goals[right_left]
        sign = 1-2*right_left
            
#        # Open hand 
#        gripper_goal.command.position = 0.9*sign
#        gripper.send_goal(gripper_goal)
#        gripper.wait_for_result(rospy.Duration(1.0))
#        time.sleep(1)
        
        #gripper pose 0.0, 0.0, sign*3.14/2
        q = quaternion_from_euler(gripper_pos[0]*sign, gripper_pos[1]*sign, gripper_pos[2]*sign)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]
            
        arm.set_pose_target(target_pose)            
        arm.go()
    
    
    def set_gripper(self, goal=0.3, right_left=0):
        gripper = self.grippers[right_left]
        gripper_goal = self.gripper_goals[right_left]
        sign = 1-2*right_left
        gripper_goal.command.position = goal*sign
        gripper.send_goal(gripper_goal)
        gripper.wait_for_result(rospy.Duration(2.0))
        
    def init_pose(self):
        self.r_arm.set_named_target("r_arm_init_pose")
        self.r_arm.go()
        self.l_arm.set_named_target("l_arm_init_pose")
        self.l_arm.go()
    
   
