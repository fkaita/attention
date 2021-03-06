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
        self._depth_sub = rospy.Subscriber(
            "/sciurus17/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback, queue_size=1)
#        self._point_depth = rospy.Subscriber("/sciurus17/camera/depth/image_rect_raw", Image, self._point_depth_callback, queue_size=1)
        self._image_pub = rospy.Publisher("~output_image", Image, queue_size=1)
        self._object_rect = [0, 0, 0, 0]
        self._object_target = [0, 0]
        self._image_shape = Point()
        self._object_detected = False
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

        # ?????????width, height?????????
        self._image_shape.x = input_image.shape[1]
        self._image_shape.y = input_image.shape[0]

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

    def get_object_position(self):
        # ???????????????0, 0??????????????????????????????????????????????????????????????????
        # ??????????????????????????????-1.0 ~ 1.0?????????????????????

        # object_center = Point(
        #        self._object_rect[0] + self._object_rect[2] * 0.5,
        #        self._object_rect[1] + self._object_rect[3] * 0.5,
        #        0)
        object_center = Point(self._object_target[0], self._object_target[0], 0)

        # ??????????????????0, 0???????????????????????????
        translated_point = Point()
        translated_point.x = object_center.x - self._image_shape.x * 0.5
        translated_point.y = -(object_center.y - self._image_shape.y * 0.5)

        # ?????????
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
                size = int(self.dist_target_size/2)
                target_depth = self.depth_img[center[0]-size:center[0]+size, center[1]-size:center[1]+size]
#                target_depth = self.round_base * np.round(target_depth/self.round_base)
#                val, count = np.unique(target_depth, return_counts=True)
#                distance_mm = val[np.argmax(count)]
                distance_mm = np.round(np.mean(target_depth[target_depth > 1]), -1)

                cv2.putText(img, "{} mm".format(distance_mm),
                            (box[0], box[1]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, self.classNames[classId-1].upper(), (box[0],
                            box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                # cv2.putText(img, str(round(confidence*100, 2)), (box[0]+150, box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('output', img)
            cv2.waitKey(1)
            self._object_detected = True
            self._object_target = (1, 1)

        return img

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
        #        object_position = object_tracker._detect_lblue_object()
#        object_position = object_tracker._detect_head_direct()


if __name__ == '__main__':
    rospy.init_node("head_camera_tracking")
    object_tracker = ObjectTracker()

    main()
