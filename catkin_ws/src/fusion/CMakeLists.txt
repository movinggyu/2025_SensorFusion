cmake_minimum_required(VERSION 3.0.2)
project(fusion)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_filters
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/fusion_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
