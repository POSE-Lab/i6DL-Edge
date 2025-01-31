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
RUN pip install tensorrt
RUN pip install plyfile
RUN apt-get update && apt-get install -y ros-noetic-rqt-image-view

COPY ./docker/launch_camera.sh .