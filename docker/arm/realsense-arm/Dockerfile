ARG TAG=latest
FROM custom-ros-arm:${TAG}
WORKDIR /tmp

# Build Realsense SDK from source
RUN  sudo apt-get update && sudo apt-get -y upgrade && sudo apt-get -y dist-upgrade
RUN  git clone https://github.com/IntelRealSense/librealsense.git
COPY ./docker/arm/realsense-arm/realsense_packages.txt ./realsense_packages.txt 
RUN  sudo xargs -a ./realsense_packages.txt apt-get install -y
RUN cd librealsense && ./scripts/setup_udev_rules.sh
RUN cd librealsense && mkdir -p build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=release \
-DBUILD_EXAMPLES=true \
-DBUILD_GRAPHICAL_EXAMPLES=true &&\
 make -j8 && sudo make -j8 install

# Explicitly use the bash interpreter, otherwise 'source' won't work
SHELL ["/bin/bash", "-c"]

# Install packages necessary for building the Realsense ROS node
RUN sudo apt-get install -y ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-diagnostic-updater

ENV HOME=/root/
# Create and initialize a catkin workspace
RUN mkdir -p ${HOME}/catkin_ws/src && \
source /opt/ros/noetic/setup.bash && catkin_init_workspace

#&& cd ~/catkin_ws/src/ &&\
#git clone https://github.com/IntelRealSense/realsense-ros.git &&\ 
#source /opt/ros/noetic/setup.bash &&\
#cd ~/catkin_ws/src/realsense-ros/ &&\
#git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1` &&\
#cd .. &&\
#sudo apt-get install -y ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-diagnostic-updater &&\
#catkin_init_workspace &&\
#cd .. && \
#catkin_make clean &&\
#catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release &&\
#catkin_make install

# RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc &&\
# source ~/.bashrc
