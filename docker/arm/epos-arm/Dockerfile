ARG TAG=latest
FROM realsense-arm:${TAG}
WORKDIR /

## Set up EPOS
# Install pip wheel for ONNX for Jetson
RUN wget https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl &&\
pip3 install --upgrade pip && pip3 install onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl

# Install necessary Python modules
COPY ./docker/arm/epos-arm/requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt


# # Build IGL from source
RUN git clone https://github.com/libigl/libigl-python-bindings.git  

# RUN python3 setup.py build --debug develop

RUN cd libigl-python-bindings && python3 -m pip install ./


RUN apt update -y && apt install -y python-is-python3
#RUN apt-get install sudo
# Directory in the *final* image where the repo will be stored
ENV REPO_PATH_SRC=.
ENV REPO_PATH_DST=/root/
#RUN mkdir ${REPO_PATH_DST}

# Copy repo from host to the specified container path 
COPY ${REPO_PATH_SRC}/epos_lib ${REPO_PATH_DST}/epos_lib
COPY ${REPO_PATH_SRC}/external ${REPO_PATH_DST}/external

ENV PYTHONPATH=$REPO_PATH_DST:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/bop_renderer/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/bop_toolkit:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/progressive-x/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/slim:$PYTHONPATH

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$REPO_PATH_DST/external/llvm/lib

# The installation locations of OSMesa and LLVM:
# ENV OSMESA_PREFIX=$REPO_PATH_DST/external/osmesa
# ENV LLVM_PREFIX=$REPO_PATH_DST/external/llvm

# RUN mkdir -p $OSMESA_PREFIX &&\
# mkdir -p $LLVM_PREFIX &&\
# cd $REPO_PATH_DST/external/bop_renderer/osmesa-install &&\
# mkdir -p ./build &&\
# cd ./build &&\
# bash ../osmesa-install.sh

# RUN cd $REPO_PATH_DST/external/bop_renderer &&\
# mkdir -p ./build &&\
# cd ./build &&\
# cmake .. -DCMAKE_BUILD_TYPE=Release &&\
# make -j8

RUN mkdir -p $REPO_PATH_DST/external/progressive-x/build &&\
cd $REPO_PATH_DST/external/progressive-x/build &&\
cmake .. -DCMAKE_BUILD_TYPE=Release &&\
make -j8
