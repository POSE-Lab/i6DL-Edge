ARG TAG=latest
FROM base-x86:${TAG}

USER root
# setup assimp
ENV ASSIMP_HEAD=a3b3ab1 

RUN cd /tmp &&\
    git clone https://github.com/assimp/assimp.git && \
    cd assimp && \
    git checkout $ASSIMP_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j8 && \
    make install 

RUN apt-get update && apt-get install -y wget 
## Setup ROS
COPY ./docker/x86/custom-ros-x86/ros_packages.txt ./ros_packages.txt 
RUN echo "deb http://packages.ros.org/ros/ubuntu ${RELEASE} main" > /etc/apt/sources.list.d/ros-latest.list
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update && xargs -a ./ros_packages.txt apt install -y

# Expose port 11311 for ROS local / remote ROS masters
EXPOSE 11311


# initialize rosdep. rosdep enables you to easily install system dependencies 
# for source you want to compile and is required to run some core components in ROS

# Also source /opt/ros/noetic/setup.bash script in every bash terminal you use ROS in.
RUN sudo rosdep init && \
    sudo rosdep update && \
    echo 'WS_SETUP_FILE=/home/$USERNAME/catkin_ws/devel/setup.bash' >> ~/.bashrc && \
    echo 'if [ -f "$WS_SETUP_FILE" ]; then' >> ~/.bashrc && \
    echo ' source $WS_SETUP_FILE' >> ~/.bashrc && \
    echo 'else' >> ~/.bashrc && \
    echo ' source /opt/ros/noetic/setup.bash' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc && \
    cd ~
