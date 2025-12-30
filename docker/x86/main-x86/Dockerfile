ARG TAG=latest
FROM epos-x86:${TAG}

USER root
WORKDIR /root


ENV BOP_PATH=/root/sandbox/bop_datasets

# Folder with TFRecord files.
# it doesn't seem to be explicitly used, however the inference script won't run
# when I remove it - TODO
ENV STORE_PATH=/root/sandbox
ENV TF_DATA_PATH=${STORE_PATH}/data  

# Images folder path
#ENV IMG_PATH=/
# Folder with trained EPOS models.
ENV TF_MODELS_PATH=${STORE_PATH}/tf_models

COPY ./docker/x86/main-x86/prepare_ws.sh .
COPY ./docker/x86/main-x86/entrypoint.sh .

RUN chmod +x ./prepare_ws.sh ./entrypoint.sh

RUN pip3 install pycuda --verbose
RUN apt-get update && apt-get -y install --fix-broken libnvinfer8=8.5.2-1+cuda11.8 \
libnvinfer-bin=8.5.2-1+cuda11.8 \
libnvinfer-plugin8=8.5.2-1+cuda11.8 \
libnvparsers8=8.5.2-1+cuda11.8 \
libnvonnxparsers8=8.5.2-1+cuda11.8 \
python3-libnvinfer=8.5.2-1+cuda11.8 \
libnvinfer-dev=8.5.2-1+cuda11.8 \
libnvinfer-plugin-dev=8.5.2-1+cuda11.8 \
libnvparsers-dev=8.5.2-1+cuda11.8 \
libnvonnxparsers-dev=8.5.2-1+cuda11.8 \
python3-libnvinfer-dev=8.5.2-1+cuda11.8
RUN pip install plyfile
RUN apt-get update && apt-get install -y ros-noetic-rqt-image-view

COPY ./docker/launch_camera.sh .