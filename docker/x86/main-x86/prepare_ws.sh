#!/bin/bash

cd ${HOME}/catkin_ws/src/
# Check if the necessary packages are installed in 
# workspace. Otherwise, install them

# Build Realsense ROS from source
# Install an older realsense version compatible with ROS noetic
p=$(rospack list-names | grep realsense2_camera)
if [[ -z $p ]]; then
    echo "Package realsense2_camera not found, installing..."
    git clone https://github.com/IntelRealSense/realsense-ros.git  
    source /opt/ros/noetic/setup.bash 
    cd ${HOME}/catkin_ws/src/realsense-ros/ 
    git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1` 
else
    echo "Package realsense2_camera found"
fi

p=$(rospack list-names | grep odl)
if [[ -z $p ]]; then
    echo "Package odl not found, installing..."
    cd ${HOME}/catkin_ws/
    catkin_make clean 
    catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release 
    catkin_make install
    #cd catkin_ws
    #source /opt/ros/noetic/setup.bash
    #catkin_make --force-cmake
    #source ${HOME}/catkin_ws/devel/setup.bash
else
    echo "Package odl found"
fi

source ~/.bashrc
