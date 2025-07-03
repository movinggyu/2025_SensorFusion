#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, PointCloud
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import random

def make_test_image(width, height):
    # 흑백 그라데이션 이미지
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        cv2.line(img, (0, i), (width-1, i), (i*255//height,)*3, 1)
    return img

def make_test_pointcloud():
    # 카메라 앞쪽 1m 지점에 100개 랜덤 포인트
    pts = np.random.uniform(-0.5, 0.5, (100,3))
    pts[:,2] += 1.0
    header = rospy.Header(frame_id="lidar_frame", stamp=rospy.Time.now())
    return pc2.create_cloud_xyz32(header, pts.tolist())

def make_lidar_bbox():
    # 랜덤 중심, 크기 생성
    center = np.array([random.uniform(-1,1), random.uniform(-1,1), random.uniform(1,3)])
    size = np.array([random.uniform(0.2,1.0), random.uniform(0.2,1.0), random.uniform(0.5,2.0)])
    dx, dy, dz = size / 2
    # 8개 코너 상대 좌표
    offsets = np.array([[ dx,  dy,  dz],
                        [ dx, -dy,  dz],
                        [-dx, -dy,  dz],
                        [-dx,  dy,  dz],
                        [ dx,  dy, -dz],
                        [ dx, -dy, -dz],
                        [-dx, -dy, -dz],
                        [-dx,  dy, -dz]])
    corners = (offsets + center).tolist()
    header = rospy.Header(frame_id="lidar_frame", stamp=rospy.Time.now())
    return pc2.create_cloud_xyz32(header, corners)

def make_yolo_bbox(width, height):
    # 랜덤 bbox 크기와 위치 생성
    w = random.randint(50, width//3)
    h = random.randint(50, height//3)
    u_min = random.randint(0, width - w)
    v_min = random.randint(0, height - h)
    uv = [
        [u_min,     v_min,     0.0],
        [u_min + w, v_min,     0.0],
        [u_min + w, v_min + h, 0.0],
        [u_min,     v_min + h, 0.0],
    ]
    header = rospy.Header(frame_id="camera_frame", stamp=rospy.Time.now())
    pts = [Point32(x=u, y=v, z=z) for u, v, z in uv]
    msg = PointCloud()
    msg.header = header
    msg.points = pts
    msg.channels = []
    return msg

if __name__ == "__main__":
    rospy.init_node("fusion_test_pub_all")
    bridge = CvBridge()

    img_pub    = rospy.Publisher("/usb_cam/image_raw", Image, queue_size=1)
    pc_pub     = rospy.Publisher("/velodyne_points", PointCloud2, queue_size=1)
    bbox3d_pub = rospy.Publisher("/lidar/bbox_3d", PointCloud2, queue_size=1)
    yolo_pub   = rospy.Publisher("/yolo/bbox", PointCloud, queue_size=1)

    rate = rospy.Rate(1)  # 1 Hz
    width, height = 640, 480

    while not rospy.is_shutdown():
        # 이미지
        img = make_test_image(width, height)
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_pub.publish(img_msg)

        # 포인트클라우드
        pc_msg = make_test_pointcloud()
        pc_pub.publish(pc_msg)

        # 라이다 BBox (랜덤)
        bbox3d_msg = make_lidar_bbox()
        bbox3d_pub.publish(bbox3d_msg)

        # YOLO BBox (랜덤)
        yolo_msg = make_yolo_bbox(width, height)
        yolo_pub.publish(yolo_msg)

        rospy.loginfo("Published image, pointcloud, random bbox3d, random yolo bbox")
        rate.sleep()
