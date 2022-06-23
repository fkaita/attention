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
import rosnode
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal,
                              FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState)

from trajectory_msgs.msg import JointTrajectoryPoint


class NeckYawPitch():
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
        self._current_yaw = 0.0  # Degree
        self._current_pitch = 0.0  # Degree

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
        # Head direct control
        # オブジェクト追跡のしきい値
        # 正規化された座標系(px, px)
        THRESH_X = 0.05
        THRESH_Y = 0.05

        # 首の初期角度 Degree
        INITIAL_YAW_ANGLE = 0
        INITIAL_PITCH_ANGLE = 0

        # 首の制御角度リミット値 Degree
        MAX_YAW_ANGLE = 120
        MIN_YAW_ANGLE = -120
        MAX_PITCH_ANGLE = 50
        MIN_PITCH_ANGLE = -70

        # 首の制御量
        # 値が大きいほど首を大きく動かす
        OPERATION_GAIN_X = 4.0  # 5
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


#def main():
#    return


#if __name__ == '__main__':

#    try:
#        if not rospy.is_shutdown():
#            main()
#    except rospy.ROSInterruptException:
#        pass
