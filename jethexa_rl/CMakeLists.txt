cmake_minimum_required(VERSION 3.0.2)
project(jethexa_rl)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  gazebo_msgs
  message_generation
)

## Generate services
add_service_files(
  FILES
  CPGControl.srv
#  Service1.srv
#  Service2.srv
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing dependencies
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES jethexa_rl
 CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs gazebo_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/jethexa_rl.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need Portable Pose Graph library by Cartographer
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(${PROJECT_NAME}_node src/jethexa_rl_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

catkin_install_python(PROGRAMS
  scripts/jethexa_gym_env.py
  scripts/train_ppo.py
  scripts/reset_world.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
) 