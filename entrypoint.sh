#!/bin/bash
set -e

# Setup ROS environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

# Set Gazebo resource path to find materials
export GAZEBO_RESOURCE_PATH="/usr/share/gazebo-11:$GAZEBO_RESOURCE_PATH"

# Set Python 3 environment variables for ROS Melodic compatibility
if [ "$ROS_DISTRO" = "melodic" ]; then
  # Make Python 3 the default
  export PYTHONPATH="/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages:$PYTHONPATH"
  # Install Python packages if not already installed
  pip3 install rospkg catkin_pkg --user
fi

# Setup the catkin workspace if it exists
if [ -f "/catkin_ws/devel/setup.bash" ]; then
  source "/catkin_ws/devel/setup.bash"
fi

# Run any provided command
exec "$@" 