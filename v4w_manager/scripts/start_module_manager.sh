#!/bin/bash
# Entry point for v4w-module-manager.service: ROS environment + this robot's module manager.
# Modules inherit this environment, so the setup file must cover every workspace
# a module launches from (new_vw_ws extends catkin_ws).
source /opt/ros/noetic/setup.bash
source "${V4W_SETUP:-$HOME/new_vw_ws/devel/setup.bash}"

ROBOT=${ROBOT_NAME:-$(hostname)}
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://192.168.0.17:11311}
export ROS_HOSTNAME=${ROS_HOSTNAME:-$(hostname).local}
# The Azure Kinect depth engine needs the desktop's X server (v4w_camera.launch sets DISPLAY=:0)
if [ -z "$XAUTHORITY" ] && [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
  export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
fi

# rosrun, not roslaunch: on master loss the manager exits and systemd restarts it
exec rosrun v4w_manager module_manager.py __ns:=/$ROBOT __name:=module_manager \
  _robot_name:=$ROBOT _config:="${V4W_MODULES:-$(rospack find v4w_manager)/config/modules.yaml}"
