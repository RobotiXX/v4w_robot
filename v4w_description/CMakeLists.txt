cmake_minimum_required(VERSION 2.8.3)
project(v4w_description)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  robot_state_publisher
  urdf
)

## Declare a catkin package
catkin_package(
  # No need to list resource directories as include directories if they contain non-code files.
  CATKIN_DEPENDS roscpp rospy std_msgs robot_state_publisher urdf
)

## Specify additional locations of header files if you have any.
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.launch"
)

## Install description resources (URDF and mesh files)
install(DIRECTORY urdf meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.*"
  PATTERN ".svn" EXCLUDE
)