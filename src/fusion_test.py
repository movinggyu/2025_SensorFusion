#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from fusion_module import Fusion

class FusionNode:
    def __init__(self):
        rospy.init_node('fusion_node')
        self.bridge = CvBridge()

        # 카메라 파라미터값 (일단 임시값)
        Fx, Fy = 600, 600
        Cx, Cy = 320, 240
        R = np.eye(3)
        T = np.zeros((3, 1))
        img_width = 640
        img_height = 480

        self.fusion = Fusion(Fx, Fy, Cx, Cy, R, T, img_width, img_height)

        # 이미지 퍼블리셔 (RViz에서 시각화할 토픽)
        self.image_pub = rospy.Publisher('/fusion/image', Image, queue_size=1)

        # 토픽 구독자 설정 (image + lidar)
        image_sub = Subscriber('/camera/image_raw', Image)
        lidar_sub = Subscriber('/lidar/points', PointCloud2)

        # 근사 시간 동기화
        ats = ApproximateTimeSynchronizer([image_sub, lidar_sub], queue_size=10, slop=0.1)
        ats.registerCallback(self.callback)

    def callback(self, image_msg, cloud_msg):
        # 카메라 이미지 → OpenCV 형식
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"CV bridge error: {e}")
            return

        # 라이다 포인트 읽기
        points = []
        for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        points_np = np.array(points)

        # 라이다 포인트를 이미지로 투영
        try:
            points_2d = self.fusion.rid2img(points_np)
        except Exception as e:
            rospy.logwarn(f"Fusion error: {e}")
            return

        # # 이미지 위에 Bbox 시각화
        # bbox = self.fusion.points2bbox(points_2d)
        # bbox = bbox.reshape((-1,1,2))

        # cv2.polylines(cv_image, [bbox], isClosed=True, color=(255, 0, 0), thickness=2) # 빨간색 박스


        # 이미지 위에 포인트 시각화
        for u, v in points_2d.astype(int):
            if 0 <= u < self.fusion.img_width and 0 <= v < self.fusion.img_height:
                cv2.circle(cv_image, (u, v), 2.5, (0, 255, 0), -1)  # 초록색 점

        # 퍼블리시할 메시지로 변환
        fusion_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        fusion_msg.header = image_msg.header  # 시간 동기화용

        self.image_pub.publish(fusion_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = FusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
