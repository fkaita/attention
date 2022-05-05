#!/usr/bin/env python
# coding: utf-8

import rospy
import math
import sys
import dlib
from imutils import face_utils
import time
import moveit_commander


# for ObjectTracker
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point

# for NeckYawPitch
import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTrajectoryControllerState
)
from trajectory_msgs.msg import JointTrajectoryPoint

class ObjectTracker:
    def __init__(self):      
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber(
            "/sciurus17/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber("/sciurus17/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback, queue_size=1)
        self._image_pub = rospy.Publisher("~output_image", Image, queue_size=1)
        self._object_rect = [0, 0, 0, 0]
        self._object_target = [0, 0]
        self._image_shape = Point()
        self._object_detected = False
        
        classFile = '/home/sciurus/Documents/data/coco.names'
        with open(classFile, 'rt') as f:
            self.classNames = [line.rstrip() for line in f]
        
        self._CV_MAJOR_VERSION, _, _ = cv2.__version__.split('.')

        
    def _image_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        # 画像のwidth, heightを取得
        self._image_shape.x = input_image.shape[1]
        self._image_shape.y = input_image.shape[0]
        
        self.input_img = input_image

        self._image_pub.publish(self._bridge.cv2_to_imgmsg(input_image, "bgr8"))
        
    def _depth_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
            
        # 画像のwidth, heightを取得
        self._image_shape.x = input_image.shape[1]
        self._image_shape.y = input_image.shape[0]
        
        
        self.depth_img = input_image

        # output_image = self._detect_object(input_image)
        # if output_image is not False:
        #     self._image_pub.publish(self._bridge.cv2_to_imgmsg(output_image, "bgr8"))

        # self._median_depth_pub.publish(self._median_depth)


    def get_object_position(self):
        # 画像中心を0, 0とした座標系におけるオブジェクトの座標を出力
        # オブジェクトの座標は-1.0 ~ 1.0に正規化される

        # object_center = Point(
        #        self._object_rect[0] + self._object_rect[2] * 0.5,
        #        self._object_rect[1] + self._object_rect[3] * 0.5,
        #        0)
        object_center = Point(self._object_target[0], self._object_target[0], 0)

        # 画像の中心を0, 0とした座標系に変換
        translated_point = Point()
        translated_point.x = object_center.x - self._image_shape.x * 0.5
        translated_point.y = -(object_center.y - self._image_shape.y * 0.5)

        # 正規化
        normalized_point = Point()
        if self._image_shape.x != 0 and self._image_shape.y != 0:
            normalized_point.x = translated_point.x / (self._image_shape.x * 0.5)
            normalized_point.y = translated_point.y / (self._image_shape.y * 0.5)

        return normalized_point

    def object_detected(self):
        return self._object_detected


    def _detect_object(self, net):
        img = self.input_img
        
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                
                center = [int((box[0]+box[2])/2), int((box[1]+box[3])/2)]
                
                distance_mm = np.array([0,0])
                for idx in range(10):
                    idx = (idx-5)*3
                    distance_mm =+ np.array(self.depth_img[center[0]+idx, center[1]+idx])
                distance_mm = distance_mm/10
                
                cv2.putText(img, "{} mm".format(distance_mm), (box[0], box[1]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, self.classNames[classId-1].upper(), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                # cv2.putText(img, str(round(confidence*100, 2)), (box[0]+150, box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('output', img)
            cv2.waitKey(1)
            self._object_detected = True
            self._object_target = (1,1)
        
        return img

def main():
    r = rospy.Rate(10)
    configPath = '/home/sciurus/Documents/data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '/home/sciurus/Documents/data/frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while not rospy.is_shutdown():
        object_position = object_tracker._detect_object(net)
    
    
    
if __name__ == '__main__':
    rospy.init_node("head_camera_tracking")
    object_tracker = ObjectTracker()
    
    main()

