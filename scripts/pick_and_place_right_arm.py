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
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState)

from trajectory_msgs.msg import JointTrajectoryPoint


def main():
    rospy.init_node("sciurus17_pick_and_place_controller")
    robot = moveit_commander.RobotCommander()
    arm = moveit_commander.MoveGroupCommander("r_arm_group")
    arm.set_max_velocity_scaling_factor(0.1)
    gripper = actionlib.SimpleActionClient("/sciurus17/controller1/right_hand_controller/gripper_cmd", GripperCommandAction)
    gripper.wait_for_server()
    gripper_goal = GripperCommandGoal()
    gripper_goal.command.max_effort = 2.0
    
    led_pub = rospy.Publisher('neopixel_simple', String , queue_size=1)

    rospy.sleep(1.0)

    print("Group names:")
    print(robot.get_group_names())

    print("Current state:")
    print(robot.get_current_state())

    # アーム初期ポーズを表示
    arm_initial_pose = arm.get_current_pose().pose
    print("Arm initial pose:")
    print(arm_initial_pose)

    # 何かを掴んでいた時のためにハンドを開く
    gripper_goal.command.position = 0.9
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # SRDFに定義されている"home"の姿勢にする
    arm.set_named_target("r_arm_init_pose")
    arm.go()
    gripper_goal.command.position = 0.0
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))
    
    

    neck = NeckYawPitch()
    
    while not neck.state_received():
        pass
    yaw_angle = neck.get_current_yaw()
    pitch_angle = neck.get_current_pitch()
    
    neck.set_angle(math.radians(0), math.radians(0), 3.0)
    
    ## Head direct control
    # オブジェクト追跡のしきい値
    # 正規化された座標系(px, px)
    THRESH_X = 0.05
    THRESH_Y = 0.05

    # 首の初期角度 Degree
    INITIAL_YAW_ANGLE = 0
    INITIAL_PITCH_ANGLE = 0

    # 首の制御角度リミット値 Degree
    MAX_YAW_ANGLE   = 120
    MIN_YAW_ANGLE   = -120
    MAX_PITCH_ANGLE = 50
    MIN_PITCH_ANGLE = -70

    # 首の制御量
    # 値が大きいほど首を大きく動かす
    OPERATION_GAIN_X = 4.0 #5
    OPERATION_GAIN_Y = 4.0

    # 初期角度に戻る時の制御角度 Degree
    RESET_OPERATION_ANGLE = 3
    
    # Horizontal and vertical visual field
    h_view = 65
    v_view = 40
    
    # 首の制御角度を制限する
    if yaw_angle > MAX_YAW_ANGLE:
        yaw_angle = MAX_YAW_ANGLE
    if yaw_angle < MIN_YAW_ANGLE:
        yaw_angle = MIN_YAW_ANGLE
        
    if pitch_angle > MAX_PITCH_ANGLE:
        pitch_angle = MAX_PITCH_ANGLE
    if pitch_angle < MIN_PITCH_ANGLE:
        pitch_angle = MIN_PITCH_ANGLE
        
    yaw_angle = -15
    pitch_angle = -15
        
    neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle),goal_secs=1.0)
    
    command = String("br")
    led_pub.publish(command)
    time.sleep(3)
    
    command = String("s")
    led_pub.publish(command)

    
    
    # 掴む準備をする
    # 0.1 is 10cm
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.5
    target_pose.position.y = -0.1
    target_pose.position.z = 0.6
#    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
    q = quaternion_from_euler(0.0, 0.0, 3.14/2)
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行
    

#    print("Current angle is... ")
#    joint_values = arm.get_current_joint_values()
#    print(np.array(joint_values)/3.14*360)
#    # 掴む準備をする
#    target_pose = geometry_msgs.msg.Pose()
#    target_pose.position.x = 0.45
#    target_pose.position.y = -0.2
#    target_pose.position.z = 0.4
##    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
#    q = quaternion_from_euler(0.0, 0.0, 3.14/2)
#    target_pose.orientation.x = q[0]
#    target_pose.orientation.y = q[1]
#    target_pose.orientation.z = q[2]
#    target_pose.orientation.w = q[3]
#    arm.set_pose_target(target_pose)  # 目標ポーズ設定
#    arm.go()  # 実行
    
    print("Current angle is... ")
    joint_values = arm.get_current_joint_values()
    print(np.array(joint_values)/3.14*360)

    # ハンドを開く
    gripper_goal.command.position = 0.7
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))
    time.sleep(2)
    
    #    # ハンドを閉じる
    gripper_goal.command.position = 0.1
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))
    
    
    ## LED GREEN
    command = String("bg")
    led_pub.publish(command)
    time.sleep(5)
    
    command = String("s")
    led_pub.publish(command)
    
    
    
    
    
class NeckYawPitch(object):
    def __init__(self):
        self.__client = actionlib.SimpleActionClient("/sciurus17/controller3/neck_controller/follow_joint_trajectory",
                                                     FollowJointTrajectoryAction)
        self.__client.wait_for_server(rospy.Duration(5.0))
        if not self.__client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Action Server Not Found")
            rospy.signal_shutdown("Action Server not found")
            sys.exit(1)

        self._state_sub = rospy.Subscriber("/sciurus17/controller3/neck_controller/state", 
                JointTrajectoryControllerState, self._state_callback, queue_size=1)

        self._state_received = False
        self._current_yaw = 0.0 # Degree
        self._current_pitch = 0.0 # Degree


    def _state_callback(self, state):
        # 首の現在角度を取得

        self._state_received = True
        yaw_radian = state.actual.positions[0]
        pitch_radian = state.actual.positions[1]

        self._current_yaw = math.degrees(yaw_radian)
        self._current_pitch = math.degrees(pitch_radian)


    def state_received(self):
        return self._state_received


    def get_current_yaw(self):
        return self._current_yaw


    def get_current_pitch(self):
        return self._current_pitch


    def set_angle(self, yaw_angle, pitch_angle, goal_secs=1.0e-2):
        # 首を指定角度に動かす
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["neck_yaw_joint", "neck_pitch_joint"]

        yawpoint = JointTrajectoryPoint()
        yawpoint.positions.append(yaw_angle)
        yawpoint.positions.append(pitch_angle)
        yawpoint.time_from_start = rospy.Duration(goal_secs)
        goal.trajectory.points.append(yawpoint)

        self.__client.send_goal(goal)
        self.__client.wait_for_result(rospy.Duration(0.1))
        return self.__client.get_result()



#    # 掴みに行く
#    target_pose = geometry_msgs.msg.Pose()
#    target_pose.position.x = 0.65
#    target_pose.position.y = 0.0
#    target_pose.position.z = 0.23
#    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
#    target_pose.orientation.x = q[0]
#    target_pose.orientation.y = q[1]
#    target_pose.orientation.z = q[2]
#    target_pose.orientation.w = q[3]
#    arm.set_pose_target(target_pose)  # 目標ポーズ設定
#    arm.go()  # 実行



#    # 掴みに行く
#    target_pose = geometry_msgs.msg.Pose()
#    target_pose.position.x = 0.25
#    target_pose.position.y = 0.0
#    target_pose.position.z = 0.13
#    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
#    target_pose.orientation.x = q[0]
#    target_pose.orientation.y = q[1]
#    target_pose.orientation.z = q[2]
#    target_pose.orientation.w = q[3]
#    arm.set_pose_target(target_pose)  # 目標ポーズ設定
#    arm.go()  # 実行

#    # ハンドを閉じる
#    gripper_goal.command.position = 0.4
#    gripper.send_goal(gripper_goal)
#    gripper.wait_for_result(rospy.Duration(1.0))

#    # 持ち上げる
#    target_pose = geometry_msgs.msg.Pose()
#    target_pose.position.x = 0.25
#    target_pose.position.y = 0.0
#    target_pose.position.z = 0.3
#    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
#    target_pose.orientation.x = q[0]
#    target_pose.orientation.y = q[1]
#    target_pose.orientation.z = q[2]
#    target_pose.orientation.w = q[3]
#    arm.set_pose_target(target_pose)  # 目標ポーズ設定
#    arm.go()							# 実行

#    # 移動する
#    target_pose = geometry_msgs.msg.Pose()
#    target_pose.position.x = 0.4
#    target_pose.position.y = 0.0
#    target_pose.position.z = 0.3
#    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
#    target_pose.orientation.x = q[0]
#    target_pose.orientation.y = q[1]
#    target_pose.orientation.z = q[2]
#    target_pose.orientation.w = q[3]
#    arm.set_pose_target(target_pose)  # 目標ポーズ設定
#    arm.go()  # 実行

#    # 下ろす
#    target_pose = geometry_msgs.msg.Pose()
#    target_pose.position.x = 0.4
#    target_pose.position.y = 0.0
#    target_pose.position.z = 0.13
#    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
#    target_pose.orientation.x = q[0]
#    target_pose.orientation.y = q[1]
#    target_pose.orientation.z = q[2]
#    target_pose.orientation.w = q[3]
#    arm.set_pose_target(target_pose)  # 目標ポーズ設定
#    arm.go()  # 実行

#    # ハンドを開く
#    gripper_goal.command.position = 0.7
#    gripper.send_goal(gripper_goal)
#    gripper.wait_for_result(rospy.Duration(1.0))

#    # 少しだけハンドを持ち上げる
#    target_pose = geometry_msgs.msg.Pose()
#    target_pose.position.x = 0.4
#    target_pose.position.y = 0.0
#    target_pose.position.z = 0.2
#    q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
#    target_pose.orientation.x = q[0]
#    target_pose.orientation.y = q[1]
#    target_pose.orientation.z = q[2]
#    target_pose.orientation.w = q[3]
#    arm.set_pose_target(target_pose)  # 目標ポーズ設定
#    arm.go()  # 実行

#    # SRDFに定義されている"home"の姿勢にする
#    arm.set_named_target("r_arm_waist_init_pose")
#    arm.go()

    print("done")


if __name__ == '__main__':

    try:
        if not rospy.is_shutdown():
            main()
    except rospy.ROSInterruptException:
        pass
