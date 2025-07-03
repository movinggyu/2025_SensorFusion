#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

def make_test_image(width, height):
    # 예: 흑백 그라데이션
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        cv2.line(img, (0, i), (width-1, i), (i*255//height,)*3, 1)
    return img

def make_test_pointcloud():
    # 예: 카메라 앞쪽 1m 지점에 100개 랜덤 포인트
    pts = np.random.uniform(-0.5, 0.5, (100,3))
    pts[:,2] += 1.0
    header = rospy.Header(frame_id="lidar_frame")
    pc2_msg = pc2.create_cloud_xyz32(header, pts.tolist())
    return pc2_msg

if __name__ == "__main__":
    rospy.init_node("fusion_test_pub")
    bridge = CvBridge()

    img_pub = rospy.Publisher("/usb_cam/image_raw", Image, queue_size=1)
    pc_pub  = rospy.Publisher("/velodyne_points", PointCloud2, queue_size=1)
    rate = rospy.Rate(1)  # 1 Hz

    width, height = 640, 480

    while not rospy.is_shutdown():
        # 이미지 생성·퍼블리시
        img = make_test_image(width, height)
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_pub.publish(img_msg)

        # 포인트클라우드 생성·퍼블리시
        pc_msg = make_test_pointcloud()
        pc_msg.header.stamp = img_msg.header.stamp
        pc_pub.publish(pc_msg)

        rospy.loginfo("-> test image & pointcloud published")
        rate.sleep()
