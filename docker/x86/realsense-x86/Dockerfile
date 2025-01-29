ARG TAG=latest
FROM custom-ros-x86:${TAG}
WORKDIR /tmp

# Build Realsense SDK from source
RUN  sudo apt-get update && sudo apt-get -y upgrade && sudo apt-get -y dist-upgrade
RUN  git clone https://github.com/IntelRealSense/librealsense.git
COPY ./docker/x86/realsense-x86/realsense_packages.txt ./realsense_packages.txt 
RUN  sudo xargs -a ./realsense_packages.txt apt-get install -y
RUN cd librealsense && ./scripts/setup_udev_rules.sh
RUN cd librealsense && mkdir -p build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=release \
-DBUILD_EXAMPLES=true \
-DBUILD_GRAPHICAL_EXAMPLES=true &&\
 make && sudo make install

SHELL ["/bin/bash", "-c"]
# Create a catkin workspace
# Build Realsense ROS from source
# Install an older realsense version compatible with ROS noetic
RUN mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src/ &&\
git clone https://github.com/IntelRealSense/realsense-ros.git &&\ 
source /opt/ros/noetic/setup.bash &&\
cd ~/catkin_ws/src/realsense-ros/ &&\
git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1` &&\
cd .. &&\
sudo apt-get install -y ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-diagnostic-updater &&\
catkin_init_workspace &&\
cd .. && \
catkin_make clean &&\
catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release &&\
catkin_make install

# RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc &&\
# source ~/.bashrc