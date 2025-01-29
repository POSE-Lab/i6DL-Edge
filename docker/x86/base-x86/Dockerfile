ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.8.0
ARG TIMEZONE=Europe/Athens
# Define the final base image
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION}
WORKDIR /
ENV USER=root
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=${TIMEZONE}
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV RELEASE=focal
# Install tools needed
# Install numpy<1.24 since some functions/data types etc are deprecated in newer numpy versions
COPY ./docker/x86/base-x86/packages.txt ./packages.txt
RUN apt-get update && xargs -a ./packages.txt apt-get install -y &&\
pip3 install --upgrade pip && pip3 uninstall numpy && pip3 install "numpy<1.24"

ENV PKG_CONFIG=/usr/bin/pkg-config

# Install Opencv 4.7.0, modified to not require 6 points for pose estimation from 3D-2D point correspondences
ARG BRANCH_NAME=4.7.0-modified
ARG OPENCV_VERSION=4.7.0
ARG OPENCV_REPO=https://github.com/opencv/opencv.git
ARG OPENCV_CONTRIB=https://github.com/opencv/opencv_contrib.git
ARG OPENCV_FORK=https://github.com/POSE-Lab/opencv_fork.git
RUN cd ./tmp && \
    git clone $OPENCV_FORK &&\
    git clone $OPENCV_CONTRIB &&\
    cd opencv_fork && \
    git checkout $BRANCH_NAME &&\
    cd ../opencv_contrib &&\
    git checkout tags/$OPENCV_VERSION &&\
    cd ../opencv_fork
RUN     mkdir -p /tmp/opencv_fork/build && \
        cd /tmp/opencv_fork/build && \
         cmake \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TIFF=ON \
          -D BUILD_JPEG=OFF \
          -D WITH_JPEG=ON \
          -D BUILD_SHARED_LIBS=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_DOCS=OFF \
          -D OpenGL_GL_PREFERENCE=GLVND \
          -D WITH_CUDA=ON \
          -D WITH_CUFFT=ON \
          -D WITH_CUBLAS=ON \
          -D BUILD_CUDA_STUBS=OFF \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=ON \
          -D CPU_DISPATCH=$CPU_DISPATCH \
          -D WITH_OPENGL=ON \
          -D WITH_QT=OFF \
          -D WITH_IPP=ON \
          -D WITH_TBB=ON \
          -D FORCE_VTK=ON \
          -D WITH_V4L=ON \
          -D WITH_XINE=ON \
          -D WITH_GDAL=ON \
          -D WITH_1394=ON \
          -D WITH_FFMPEG=ON \
          -D WITH_GSTREAMER=ON \
          -D WITH_OPENMP=ON \
          -D WITH_OPENNI=OFF \
          -D WITH_ITT=OFF \
          -DWITH_GTK_2_X=ON \
          -D BUILD_PROTOBUF=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_dnn=OFF \
          -D BUILD_opencv_sfm=OFF \
          -D BUILD_opencv_datasets=OFF \
          -D BUILD_opencv_bioinspired=OFF \
          -D BUILD_opencv_bgsegm=OFF \
          -D BUILD_opencv_hfs=OFF \
          -D BUILD_opencv_face=OFF \
          -D BUILD_opencv_freetype=OFF \
          -D BUILD_opencv_text=OFF \
          -D BUILD_opencv_ml=OFF \
          -D BUILD_opencv_surface_matching=OFF \
          -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
           -D OPENCV_PYTHON3_INSTALL_PATH=/usr/lib/python3/dist-packages \
          -D CMAKE_BUILD_TYPE=Release \
          -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules .. &&\
    	    make -j8 install &&\
   	   rm -r /tmp/opencv_fork && \
    	 rm -r /tmp/opencv_contrib

