cmake_minimum_required(VERSION 2.8.3)
project(v4w_description)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  robot_state_publisher
  urdf
  xacro
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS robot_state_publisher urdf xacro
)

## Specify additional locations of header files if you have any.
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install description resources (URDF and mesh files)
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
  FILES_MATCHING 
  PATTERN "*.urdf"
  PATTERN "*.xacro"
)

install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
  FILES_MATCHING 
  PATTERN "*.stl"
  PATTERN "*.obj"
  PATTERN "*.mtl"
)