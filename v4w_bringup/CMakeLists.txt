cmake_minimum_required(VERSION 2.8.3)
project(v4w_bringup)

find_package(catkin REQUIRED)

catkin_package()

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
