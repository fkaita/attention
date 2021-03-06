#!/usr/bin/env python3
# coding: utf-8

import rospy
import sys
import dlib
from imutils import face_utils
import time
import moveit_commander

# for ObjectTracker
import cv2
import numpy as np
import random
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point


import math
import geometry_msgs.msg
import rosnode
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal,
                              FollowJointTrajectoryAction, FollowJointTrajectoryGoal, 
                              JointTrajectoryControllerState)
from trajectory_msgs.msg import JointTrajectoryPoint

#for led control
from std_msgs.msg import String, Float64

# for speech
import japanese_text_to_speech.msg

class ObjectTracker:
    def __init__(self):
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber(
            "/sciurus17/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber(
            "/sciurus17/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback, queue_size=1)
#        self._point_depth = rospy.Subscriber("/sciurus17/camera/depth/image_rect_raw", Image, self._point_depth_callback, queue_size=1)
        self._image_pub = rospy.Publisher("~output_image", Image, queue_size=1)
        self._object_rect = [0, 0, 0, 0]
        self._object_target = [0, 0]
        self._image_shape = Point()
        self._object_detected = False
        self._face_detected = False
        self.dist_target_size = 20  # Estimate distance by mean of size x size pixcels
        self.round_base = 20  # round by this number in mm
        self._face_cascade = cv2.CascadeClassifier(
            "/home/sciurus/Documents/data/haarcascades/haarcascade_frontalface_default.xml")
        self._eyes_cascade = cv2.CascadeClassifier(
            "/home/sciurus/Documents/data/haarcascades/haarcascade_eye.xml")

        classFile = '/home/sciurus/Documents/data/coco.names'
        with open(classFile, 'rt') as f:
            self.classNames = [line.rstrip() for line in f]

        self._CV_MAJOR_VERSION, _, _ = cv2.__version__.split('.')

    def _image_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        # ?????????width, height?????????
        self._image_shape.x = input_image.shape[0]
        self._image_shape.y = input_image.shape[1]

        self.input_img = input_image

        self._image_pub.publish(self._bridge.cv2_to_imgmsg(input_image, "bgr8"))

    def _depth_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # ?????????width, height?????????
#        self._image_shape.x = input_image.shape[1]
#        self._image_shape.y = input_image.shape[0]

        self.depth_img = np.array(input_image)

        # output_image = self._detect_object(input_image)
        # if output_image is not False:
        #     self._image_pub.publish(self._bridge.cv2_to_imgmsg(output_image, "bgr8"))

        # self._median_depth_pub.publish(self._median_depth)

#    def _point_depth_callback(self, ros_image):
#        try:
#            input_image = self._bridge.imgmsg_to_cv2(ros_image, "passthrough")
#        except CvBridgeError as e:
#            rospy.logerr(e)
#            return
#
#        self.point_depth_img = np.array(input_image)
#        print(self.point_depth_img[100,100])

    def get_object_position(self, face=False):
        # ???????????????0, 0??????????????????????????????????????????????????????????????????
        # ??????????????????????????????-1.0 ~ 1.0?????????????????????

        if face:
            object_center = Point(
                self._face_rect[0] + self._face_rect[2] * 0.5,
                self._face_rect[1] + self._face_rect[3] * 0.5,
                0)
        else:
            object_center = Point(
                self._object_rect[0] + self._object_rect[2] * 0.5,
                self._object_rect[1] + self._object_rect[3] * 0.5,
                0)

        # ??????????????????0, 0???????????????????????????
        translated_point = Point()
        translated_point.x = object_center.x - self._image_shape.x * 0.5
        translated_point.y = -(object_center.y - self._image_shape.y * 0.5)
        

        # ?????????
        normalized_point = Point()
        if self._image_shape.x != 0 and self._image_shape.y != 0:
            normalized_point.x = translated_point.x / (self._image_shape.x * 0.5)
            normalized_point.y = translated_point.y / (self._image_shape.y * 0.5)
            
#        print(normalized_point.x, normalized_point.y)

        return normalized_point

    def object_detected(self):
        return self._object_detected
        
    def face_detected(self):
        return self._face_detected

    def _detect_object(self, net, object_id=52):
        # class_id 52 is banana
        img = self.input_img

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        
        self._object_detected = False

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, self.classNames[classId-1].upper(), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                if classId == object_id:
                    center = [int((box[0]+box[2])/2), int((box[1]+box[3])/2)]
                    size = int(self.dist_target_size/2)
                    target_depth = self.depth_img[center[0] -
                                                  size:center[0]+size, center[1]-size:center[1]+size]
#                target_depth = self.round_base * np.round(target_depth/self.round_base)
#                val, count = np.unique(target_depth, return_counts=True)
#                distance_mm = val[np.argmax(count)]
                    self.distance_cm = np.round(np.mean(target_depth[target_depth > 1]), -1)/10

                    cv2.putText(img, "{} cm".format(self.distance_cm),
                                (box[0], box[1]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # cv2.putText(img, str(round(confidence*100, 2)), (box[0]+150, box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                
                    self._object_detected = True
                    
                    self._object_rect = [min(box[0],box[2]),min(box[1],box[3]),math.fabs(box[2]-box[0]),math.fabs(box[3]-box[1])]
            
                
                    
                cv2.imshow('output', img)
                cv2.waitKey(1)
        return 

    def _detect_color_object(self, bgr_image, lower_color, upper_color):
        # ??????????????????????????????????????????????????????

        MIN_OBJECT_SIZE = 1000  # px * px

        # BGR?????????HSV??????????????????
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # ????????????????????????????????????
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # ??????????????????????????????
        if self._CV_MAJOR_VERSION == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ????????????????????????????????????????????????
        rects = []
        for contour in contours:
            approx = cv2.convexHull(contour)
            rect = cv2.boundingRect(approx)
            rects.append(rect)

        self._object_detected = False
        if len(rects) > 0:
            # ?????????????????????????????????
            rect = max(rects, key=(lambda x: x[2] * x[3]))

            # ???????????????????????????????????????????????????
            if rect[2] * rect[3] > MIN_OBJECT_SIZE:
                # ?????????????????????????????????????????????
                cv2.rectangle(bgr_image,
                              (rect[0], rect[1]),
                              (rect[0] + rect[2], rect[1] + rect[3]),
                              (0, 0, 255), thickness=2)

                cv2.imshow('output', bgr_image)
                cv2.waitKey(1)

                self._object_rect = rect
                self._object_detected = True

        return bgr_image

    def _detect_lblue_object(self):
        img = self.input_img
        # H: 0 ~ 179 (0 ~ 360??)
        # S: 0 ~ 255 (0 ~ 100%)
        # V: 0 ~ 255 (0 ~ 100%)
        lower_lblue = np.array([0, 130, 200])
        upper_lblue = np.array([150, 220, 255])

        return self._detect_color_object(img, lower_lblue, upper_lblue)
        
    def _detect_face(self):
        # ???????????????(??????)???????????????
        
        bgr_image = self.input_img

        SCALE = 1

        # ??????????????????????????????????????????????????????????????????
        if self._face_cascade == "" or self._eyes_cascade == "":
            rospy.logerr("cascade file does not set")
            return

        # BGR?????????????????????????????????
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # ??????????????????????????????????????????
        height, width = gray.shape[:2]
        small_gray = cv2.resize(gray, (int(width/SCALE), int(height/SCALE)))

        # ????????????????????????????????????????????????
        small_faces = self._face_cascade.detectMultiScale(small_gray)

        self._object_detected = False
        for small_face in small_faces:
            # ???????????????????????????????????????
            face = small_face*SCALE

            # ???????????????????????????????????????
            roi_gray = gray[
                face[1]:face[1]+face[3],
                face[0]:face[0]+face[2]]

            # ???????????????????????????
            eyes = self._eyes_cascade.detectMultiScale(roi_gray)

            # ?????????????????????????????????rect(??????????????????)???????????????
            if len(eyes) > 0:
                cv2.rectangle(bgr_image,
                              (face[0], face[1]),
                              (face[0]+face[2], face[1]+face[3]),
                              (0, 0, 255), 2)

                self._face_rect = face
                self._face_detected = True
                break

        return

    def _detect_head_direct(self):

        bgr_image = self.input_img

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            '/home/sciurus/Documents/data/shape_predictor_68_face_landmarks.dat')

        # BGR?????????????????????????????????
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        SCALE = 2

        # ??????????????????????????????????????????
        height, width = gray.shape[:2]
        small_gray = cv2.resize(gray, (int(width/SCALE), int(height/SCALE)))

        rects = detector(small_gray, 0)

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        size = bgr_image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [
            0, focal_length, center[1]], [0, 0, 1]], dtype="double")

        self._object_detected = False

        for rect in rects:
            shape0 = predictor(small_gray, rect)
            shape0 = np.array(face_utils.shape_to_np(shape0))

        if len(rects) > 0:
            image_points = np.array([
                (shape0[30, :]),  # nose tip
                (shape0[8, :]),  # Chin
                (shape0[36, :]),  # Left eye left corner
                (shape0[45, :]),  # right eye right corner
                (shape0[48, :]),  # left mouth corner
                (shape0[54, :])  # right mouth corner
            ], dtype='double')
            image_points = image_points*SCALE

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                          image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            (nose_end_point2D, jacobian) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(gray, p1, p2, (255, 0, 0), 2)
            try:
                cv2.imshow('output', gray)
                cv2.waitKey(1)
            except Exception as err:
                print(err)

            self._object_target = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            self._object_detected = True

        return bgr_image

    def head2base_position(self, sx, sy, pitch, yaw, waist):
    
        l = 30 #self.distance_cm
        # l is distance in cm, sx, sy are 0-1 scale.
        # p, y, w are of pitch, yaw, waist
             
        x0 = l/np.sqrt(1 + 0.13*(sx**2) + 0.41*(sy**2))
        x = np.array([x0, 0.64*sx*x0, 0.36*sy*x0, 1])

        # Define external variables, distance in cm
        p1 = np.array([[0, 0, 14.5]])
        r1 = np.array([[np.cos(waist), -np.sin(waist), 0],
                      [np.sin(waist), np.cos(waist), 0],
                      [0, 0, 1]])
        p2 = np.array([[8, 0, 33.5]])
        r2 = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
        p3 = np.array([[0, 0, 4]])
        r3 = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])
        p4 = np.array([[7, 3.5, 10]])
        e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        zeros = np.array([[0, 0, 0]])

        v1 = np.concatenate((e, p1.T), axis=1)
        v2 = np.concatenate((r1, zeros.T), axis=1)
        v3 = np.concatenate((e, p2.T), axis=1)
        v4 = np.concatenate((r2, zeros.T), axis=1)
        v5 = np.concatenate((e, p3.T), axis=1)
        v6 = np.concatenate((r3, zeros.T), axis=1)
        v7 = np.concatenate((e, p4.T), axis=1)

        for v in [v7, v6, v5, v4, v3, v2, v1]:
            v = np.concatenate((v, np.array([[0, 0, 0, 1]])))
            x = np.dot(v, x)

        return np.around(x[:3]/100, decimals=1) # x is np.array base axis x,y,z in meter



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
        # ???????????????????????????

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


    def set_angle(self, yaw_angle, pitch_angle, goal_secs=1.0e-9):
        # ??????????????????????????????
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
        

    def set_hand(self, hand_pos, gripper_pos, right_left=0, goal_secs=1.0e-9):
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
            
        # Open hand 
        gripper_goal.command.position = 0.9*sign
        gripper.send_goal(gripper_goal)
        gripper.wait_for_result(rospy.Duration(1.0))
        time.sleep(1)
        
        #gripper pose 0.0, 0.0, sign*3.14/2
        q = quaternion_from_euler(gripper_pos[0], gripper_pos[1], gripper_pos[2]*sign)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]
            
        arm.set_pose_target(target_pose)            
        arm.go()
    
    
    def set_gripper(self, goal=0.3, right_left=0, goal_secs=1.0e-9):
        gripper = self.grippers[right_left]
        gripper_goal = self.gripper_goals[right_left]
        sign = 1-2*right_left
        gripper_goal.command.position = goal*sign
        gripper.send_goal(gripper_goal)
        gripper.wait_for_result(rospy.Duration(2.0))


def hook_shutdown():
    # shutdown??????0????????????
    neck.set_angle(math.radians(0), math.radians(0), 3.0)


def main():
    rate = 20
    r = rospy.Rate(rate)
    rospy.on_shutdown(hook_shutdown)
      
    # Path for object detection
    configPath = '/home/sciurus/Documents/data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '/home/sciurus/Documents/data/frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Code for led
    led_pub = rospy.Publisher('neopixel_simple', String , queue_size=1)
    
    # Code for speech
    action_name = rospy.get_param('~action_name', 'japanese_text_to_speech')
    speed_rate = rospy.get_param('~speed_rate', 0.5)
    simple_client = actionlib.SimpleActionClient(action_name, japanese_text_to_speech.msg.SpeakAction)
            
    rospy.loginfo('Waiting japanese_text_to_speech server')
    simple_client.wait_for_server()
    
    # Set init pose open and close hand
    for i in range(2):
        handgripper.set_gripper(goal=0.9, right_left=i)
        arm = handgripper.arms[i]      
        # SRDF????????????????????????"home"??????????????????
        arm.set_named_target(handgripper.arm_poses[i])
        arm.go()
        handgripper.set_gripper(goal=0.0, right_left=i)
        
    # ???????????????????????????????????????
    # ???????????????????????????(px, px)
    THRESH_X = 0.05
    THRESH_Y = 0.05

    # ?????????????????? Degree
    INITIAL_YAW_ANGLE = 0
    INITIAL_PITCH_ANGLE = 0

    # ????????????????????????????????? Degree
    MAX_YAW_ANGLE = 120
    MIN_YAW_ANGLE = -120
    MAX_PITCH_ANGLE = 50
    MIN_PITCH_ANGLE = -70

    # ???????????????
    # ?????????????????????????????????????????????
    OPERATION_GAIN_X = 8.0  # 5
    OPERATION_GAIN_Y = 8.0

    # ??????????????????????????????????????? Degree
    RESET_OPERATION_ANGLE = 3
    
    # Horizontal and vertical visual field
    h_view = 65
    v_view = 40
    
    while not neck.state_received():
        pass
    yaw_angle = neck.get_current_yaw()
    pitch_angle = neck.get_current_pitch()
    
    detection_timestamp = rospy.Time.now()
    look_object = False
    find_object = False
    
    while not rospy.is_shutdown():
    
        yaw_angle = neck.get_current_yaw()
        pitch_angle = neck.get_current_pitch()
  
	# Object detection
        object_tracker._detect_object(net)       

        if object_tracker.object_detected():
            find_object = True
            position_map = []
            past_time = 0

            while past_time < 1.0:
                edges = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
                object_position = object_tracker.get_object_position()
                position_map.append([object_position.x, object_position.y])
                past_time = (rospy.Time.now() - detection_timestamp).to_sec()
                object_tracker._detect_object(net)

            H, _, _ = np.histogram2d(np.array(position_map).T[0], np.array(
                position_map).T[1], bins=(edges, edges))
            
            target_loc = np.where(H == H.max())
#            print(object_position.x, object_position.y)
#            print(target_loc)
            if (target_loc[0] in range(3, 7)) and (target_loc[1] in range(3, 7)):
                look_object = False  # the object locates around the center
            else:
                look_object = True
                target_position_x = (target_loc[0] - 4.5)/4.5
                target_position_y = (target_loc[1] - 4.5)/4.5
                
            detection_timestamp = rospy.Time.now()
                
        else:
            lost_time = rospy.Time.now() - detection_timestamp
            # ?????????????????????????????????????????????????????????????????????????????????
            if lost_time.to_sec() > 2.0:
                find_object = False

        if find_object and look_object:
            # ??????????????????????????????????????????????????????????????????
            # CANNOT LOOK AT OBJECT IN RIGHT SIDE
            if math.fabs(target_position_x) > THRESH_X:
                yaw_angle += - 0.6 * \
                    math.degrees(math.atan(target_position_x * 2 *
                                 math.tan(math.radians(h_view/2))))

            if math.fabs(target_position_y) > THRESH_Y:
                pitch_angle += 0.6 * \
                    math.degrees(math.atan(target_position_y * 2 *
                                 math.tan(math.radians(v_view/2))))
                                 
            # ?????????????????????????????????
            if yaw_angle > MAX_YAW_ANGLE:
                yaw_angle = MAX_YAW_ANGLE
            if yaw_angle < MIN_YAW_ANGLE:
                yaw_angle = MIN_YAW_ANGLE

            if pitch_angle > MAX_PITCH_ANGLE:
                pitch_angle = MAX_PITCH_ANGLE
            if pitch_angle < MIN_PITCH_ANGLE:
                pitch_angle = MIN_PITCH_ANGLE                
                
            neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle),goal_secs=2.0)
            time.sleep(1)
                
        elif find_object:
        
            # Green LED
            command = String("sg")
            led_pub.publish(command)
            
            #Speech
            text = rospy.get_param('~text', '?????????')
            rospy.loginfo('Sending goal to server')
            speech_goal = japanese_text_to_speech.msg.SpeakGoal()
            speech_goal.text = text
            speech_goal.speed_rate = speed_rate
            simple_client.send_goal_and_wait(speech_goal)  
            
            rospy.sleep(3)
            yaw_angle = neck.get_current_yaw()
            pitch_angle = neck.get_current_pitch()
            print(object_position.x, object_position.y)
            
            ## Reaching closer hand
            object_position = object_tracker.get_object_position()
            sx = object_position.x
            sy = object_position.y
            pitch = math.radians(pitch_angle)
            yaw = math.radians(yaw_angle)
            waist = 0
            base_pos = object_tracker.head2base_position(sx, sy, pitch, yaw, waist)
            print(pitch, yaw)            
            
            # Look at person
            neck.set_angle(math.radians(0), math.radians(15), goal_secs=1.0)
            rospy.sleep(2)
            
            # Look at object
            neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle), goal_secs=1.0)
            rospy.sleep(2)
            
            # reach hand
            if base_pos[1] <= 0:
                right_left = 0
            else:
                right_left = 1
                
            gripper_pos = [0.0, 0.0, 3.14/2]
            handgripper.set_hand(base_pos, gripper_pos, right_left=right_left)
            time.sleep(1)
            
            # Close hand
            handgripper.set_gripper(goal=0.2, right_left=right_left)
            rospy.sleep(2)
            
            # Check if object in hand or not
            hand_currents = []
            for j in range(10):
                current = handgripper.get_hand_current()[right_left]
                hand_currents.append(current)
            
            mean_current = np.mean(hand_currents)
            print('mean current is ', mean_current)
            
            if np.absolute(mean_current) > 50:
                text = rospy.get_param('~text', '???????????????????????????????????????')
                rospy.loginfo('Sending goal to server')
                speech_goal = japanese_text_to_speech.msg.SpeakGoal()
                speech_goal.text = text
                speech_goal.speed_rate = speed_rate
                simple_client.send_goal_and_wait(speech_goal)  
            
           
            
            # init pose
            arm = handgripper.arms[right_left]
            arm.set_named_target(handgripper.arm_poses[right_left])
            arm.go()      
            
            print('done')
            command = String("s")
            led_pub.publish(command)
	    

        else:
            # ?????????????????????????????????
            yaw_angle = 0
            pitch_angle = 0
            neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle),goal_secs=3.0)

        r.sleep()
               
        # pattern
        # find object
        ## object in center -> reach hand
        ## object in edge -> look at object
        # not find object -> back to center after 2 sec 
        ## after a while search object    
        


if __name__ == '__main__':
    rospy.init_node("head_camera_tracking")
    neck = NeckYawPitch()
    handgripper = HandGripper()
    object_tracker = ObjectTracker()

    main()
