#!/usr/bin/env python
# coding: utf-8

import rospy
import math
import sys
import dlib
from imutils import face_utils
import time
import moveit_commander
from neck_yaw_pitch import NeckYawPitch


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
        # 画像中心を0, 0とした座標系におけるオブジェクトの座標を出力
        # オブジェクトの座標は-1.0 ~ 1.0に正規化される

        # object_center = Point(
        #        self._object_rect[0] + self._object_rect[2] * 0.5,
        #        self._object_rect[1] + self._object_rect[3] * 0.5,
        #        0)
        object_center = Point(self._object_target[0], self._object_target[1], 0)

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

    def _detect_object(self, net, object_id=[51]):
        # class_id 51 is banana
        img = self.input_img

        classIds, confs, bbox = net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

                if classId in object_id:
                    center = [int((box[0]+box[2])/2), int((box[1]+box[3])/2)]
                    size = int(self.dist_target_size/2)
                    target_depth = self.depth_img[center[0] -
                                                  size:center[0]+size, center[1]-size:center[1]+size]
#                target_depth = self.round_base * np.round(target_depth/self.round_base)
#                val, count = np.unique(target_depth, return_counts=True)
#                distance_mm = val[np.argmax(count)]
                    distance_cm = np.round(np.mean(target_depth[target_depth > 1]), -1)/10

                    cv2.putText(img, "{} cm".format(distance_cm),
                                (box[0], box[1]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, self.classNames[classId-1].upper(), (box[0],
                                box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    # cv2.putText(img, str(round(confidence*100, 2)), (box[0]+150, box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow('output', img)
                    cv2.waitKey(1)
                    self._object_detected = True

                    object_center = Point(center[0], center[1], 0)

                    # 画像の中心を0, 0とした座標系に変換
                    translated_point = Point()
                    translated_point.x = object_center.x - self._image_shape.x * 0.5
                    translated_point.y = -(object_center.y - self._image_shape.y * 0.5)

                    # 正規化
                    normalized_point = Point()
                    if self._image_shape.x != 0 and self._image_shape.y != 0:
                        normalized_point.x = translated_point.x / (self._image_shape.x * 0.5)
                        normalized_point.y = translated_point.y / (self._image_shape.y * 0.5)

                    self._object_target = [normalized_point.x, normalized_point.y, distance_cm]
        return

    def _detect_color_object(self, bgr_image, lower_color, upper_color):
        # 画像から指定された色の物体を検出する

        MIN_OBJECT_SIZE = 1000  # px * px

        # BGR画像をHSV色空間に変換
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # 色を抽出するマスクを生成
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # マスクから輪郭を抽出
        if self._CV_MAJOR_VERSION == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭を長方形に変換し、配列に格納
        rects = []
        for contour in contours:
            approx = cv2.convexHull(contour)
            rect = cv2.boundingRect(approx)
            rects.append(rect)

        self._object_detected = False
        if len(rects) > 0:
            # 最も大きい長方形を抽出
            rect = max(rects, key=(lambda x: x[2] * x[3]))

            # 長方形が小さければ検出判定にしない
            if rect[2] * rect[3] > MIN_OBJECT_SIZE:
                # 抽出した長方形を画像に描画する
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
        # H: 0 ~ 179 (0 ~ 360°)
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

        # BGR画像をグレー画像に変換
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        SCALE = 2

        # 処理時間短縮のため画像を縮小
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

    def head2base_position(sx, sy, l, p=0, y=0, w=0):
        # l is distance in cm, sx, sy are 0-1 scale.
        # p, y, w are of pitch, yaw, waist
        x0 = l/np.sqrt(1 + 0.13*(sx**2) + 0.41*(sy**2))
        x = np.array([x0, 0.64*sx*x0, 0.36*sy*x0, 1])

        # Define external variables, distance in cm
        p1 = np.array([[0, 0, 14.5]])
        r1 = np.array([[np.cos(w), -np.sin(w), 0],
                      [np.sin(w), np.cos(w), 0],
                      [0, 0, 1]])
        p2 = np.array([[8, 0, 33.5]])
        r2 = np.array([[np.cos(y), -np.sin(y), 0],
                      [np.sin(y), np.cos(y), 0],
                      [0, 0, 1]])
        p3 = np.array([[0, 0, 4]])
        r3 = np.array([[np.cos(p), 0, np.sin(p)],
                      [0, 1, 0],
                      [-np.sin(p), 0, np.cos(p)]])
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

        return x


def main():
    r = rospy.Rate(10)
    rospy.on_shutdown(hook_shutdown)

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
    OPERATION_GAIN_X = 8.0  # 5
    OPERATION_GAIN_Y = 8.0

    # 初期角度に戻る時の制御角度 Degree
    RESET_OPERATION_ANGLE = 3

    # Path for object detection
    configPath = '/home/sciurus/Documents/data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '/home/sciurus/Documents/data/frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while not rospy.is_shutdown():
        object_tracker._detect_object(net)

        if object_tracker.object_detected():
            detection_timestamp = rospy.Time.now()
            look_object = True
        else:
            lost_time = rospy.Time.now() - detection_timestamp
            # 一定時間オブジェクトが見つからない場合は初期角度に戻る
            if lost_time.to_sec() > 1.0:
                look_object = False

        if look_object:
            # オブジェクトが画像中心にくるように首を動かす
            if np.absolute(object_tracker._object_target[0]) > THRESH_X:
                yaw_angle += -object_tracker._object_target[0] * OPERATION_GAIN_X

            if np.absolute(object_tracker._object_target[1]) > THRESH_Y:
                pitch_angle += object_tracker._object_target[1] * OPERATION_GAIN_Y

            # while not neck.state_received():
            #     pass
            # yaw_angle = neck.get_current_yaw()
            # pitch_angle = neck.get_current_pitch()
            #
            # object_position = head2base_position(
            #     object_tracker._object_target[0], object_tracker._object_target[1], object_tracker._object_target[2], p=pitch_angle, y=yaw_angle, w=0)
            #
            # x = object_position[0]/100
            # y = object_position[1]/100
            # z = object_position[2]/100

                # 首の制御角度を制限する
            if yaw_angle > MAX_YAW_ANGLE:
                yaw_angle = MAX_YAW_ANGLE
            if yaw_angle < MIN_YAW_ANGLE:
                yaw_angle = MIN_YAW_ANGLE

            if pitch_angle > MAX_PITCH_ANGLE:
                pitch_angle = MAX_PITCH_ANGLE
            if pitch_angle < MIN_PITCH_ANGLE:
                pitch_angle = MIN_PITCH_ANGLE

        else:
            # ゆっくり初期角度へ戻る
            diff_yaw_angle = yaw_angle - INITIAL_YAW_ANGLE
            if np.deg2rad(diff_yaw_angle) > RESET_OPERATION_ANGLE:
                yaw_angle -= np.copysign(RESET_OPERATION_ANGLE, diff_yaw_angle)
            else:
                yaw_angle = INITIAL_YAW_ANGLE

            diff_pitch_angle = pitch_angle - INITIAL_PITCH_ANGLE
            if np.deg2rad(diff_pitch_angle) > RESET_OPERATION_ANGLE:
                pitch_angle -= np.copysign(RESET_OPERATION_ANGLE, diff_pitch_angle)
            else:
                pitch_angle = INITIAL_PITCH_ANGLE

        neck.set_angle(np.deg2rad(yaw_angle), np.deg2rad(pitch_angle))

        r.sleep()


if __name__ == '__main__':
    rospy.init_node("head_camera_tracking")
    neck = NeckYawPitch()
    object_tracker = ObjectTracker()

    main()
