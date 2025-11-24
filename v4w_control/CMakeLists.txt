cmake_minimum_required(VERSION 2.8.3)
project(v4w_control)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf2_ros
)

## Declare a catkin package, listing dependencies as needed.
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs nav_msgs sensor_msgs geometry_msgs tf2_ros
)

## Specify include directories (if you have header files, they would be added here)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install launch files:
## This command copies the entire 'launch' directory into the package share directory.
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install config files:
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
  PATTERN "*.yml"
)

## Install Python scripts:
catkin_install_python(PROGRAMS 
  scripts/control.py
  scripts/imu_processor.py
  scripts/odometry.py
  scripts/utils.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)