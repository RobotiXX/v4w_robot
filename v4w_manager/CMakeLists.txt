cmake_minimum_required(VERSION 3.0.2)
project(v4w_manager)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

add_message_files(FILES ModuleState.msg ModuleStatus.msg)
add_service_files(FILES ModuleCommand.srv ModuleLog.srv)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

catkin_install_python(PROGRAMS scripts/module_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(PROGRAMS scripts/start_module_manager.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch systemd
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
