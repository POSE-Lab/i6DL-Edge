ARG TAG=latest
FROM epos-arm:${TAG}

ENV BOP_PATH=/root/sandbox/bop_datasets

# Folder with TFRecord files.
# it doesn't seem to be explicitly used, however the inference script won't run
# when I remove it - TODO
ENV STORE_PATH=/root/sandbox
ENV TF_DATA_PATH=${STORE_PATH}/data  

# Images folder path
#ENV IMG_PATH=/
# Folder with trained EPOS models.
ENV TF_MODELS_PATH=${STORE_PATH}/tf_models/models 
#RUN mkdir -p ${TF_MODELS_PATH}
#ENV EPOS_MODEL_NAME=crfAndLab12 
#COPY ./models ${TF_MODELS_PATH}
#ENV ONNX_MODEL_PATH=${TF_MODELS_PATH}/${EPOS_MODEL_NAME}/crfAndLab12_640_480.onnx
#COPY odl ${HOME}/catkin_ws/src
#ENV ROS_MASTER_URI=http://192.168.9.2:11311/
#ENV ROS_IP=192.168.9.4
#ENV ROS_HOSTNAME=192.168.9.4
COPY ./docker/arm/main-arm/prepare_ws.sh ${HOME}
COPY ./docker/arm/main-arm/entrypoint.sh ${HOME}

WORKDIR ${HOME}
RUN chmod +x ./prepare_ws.sh ./entrypoint.sh

RUN pip install PyOpenGL PyOpenGL_accelerate
COPY ./docker/arm/main-arm/renderings ${HOME}/renderings
RUN apt-get install -y tzdata
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone\
dpkg-reconfigure -f noninteractive tzdata

RUN apt-get install cuda-nvcc-11-4
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PATH=/usr/local/cuda-11.4/bin:$PATH
RUN pip3 install pycuda --verbose

COPY ./docker/arm/main-arm/test_pycuda.py ${HOME}
RUN echo "deb https://repo.download.nvidia.com/jetson/common r35.2 main" >> /etc/apt/sources.list && \
apt-get update && apt-get -y install python3-libnvinfer-dev libnvinfer-bin
#RUN export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
RUN pip install plyfile
