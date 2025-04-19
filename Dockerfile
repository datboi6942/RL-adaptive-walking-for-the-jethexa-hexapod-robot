FROM ros:melodic-robot

# Install Python 3 essentials first
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-yaml \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Install ROS packages
RUN apt-get update && apt-get install -y \
    ros-melodic-gazebo-ros-pkgs \
    ros-melodic-gazebo-ros-control \
    ros-melodic-effort-controllers \
    ros-melodic-position-controllers \
    ros-melodic-joint-state-controller \
    ros-melodic-joint-state-publisher \
    ros-melodic-joint-state-publisher-gui \
    ros-melodic-robot-state-publisher \
    ros-melodic-rviz \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install catkin tools (Python 2 version only)
RUN apt-get update && apt-get install -y \
    python-catkin-tools \
    python-osrf-pycommon \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3 packages for RL with specific versions for compatibility
RUN pip3 install \
    gym==0.21.0 \
    protobuf==3.19.6 \
    torch==1.10.1 \
    pyyaml==5.3.1 \
    rospkg \
    catkin_pkg \
    && pip3 install stable-baselines3==1.3.0 \
    && pip3 install tensorboard==2.6.0

# Setup catkin workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Init catkin workspace
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin init"

# Set environment variables
ENV PYTHONPATH="/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages:${PYTHONPATH}"

# Entry point for container
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"] 