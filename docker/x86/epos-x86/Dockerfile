ARG TAG=latest
FROM realsense-x86:${TAG}
WORKDIR /

## Set up EPOS
# Install necessary Python modules
COPY ./docker/x86/epos-x86/requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt


RUN sudo git clone https://github.com/libigl/libigl-python-bindings.git &&\
cd libigl-python-bindings &&\ 
python3 -m pip install libigl

# Directory in the *final* image where the repo will be stored
ENV REPO_PATH_SRC=.
ENV REPO_PATH_DST=/root/

COPY ${REPO_PATH_SRC}/epos_lib ${REPO_PATH_DST}/epos_lib
COPY ${REPO_PATH_SRC}/external ${REPO_PATH_DST}/external

ENV PYTHONPATH=$REPO_PATH_DST:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/bop_renderer/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/bop_toolkit:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/progressive-x/build:$PYTHONPATH
ENV PYTHONPATH=$REPO_PATH_DST/external/slim:$PYTHONPATH

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$REPO_PATH_DST/external/llvm/lib

RUN sudo mkdir -p $REPO_PATH_DST/external/progressive-x/build &&\
cd $REPO_PATH_DST/external/progressive-x/build &&\
sudo cmake .. -DCMAKE_BUILD_TYPE=Release &&\
sudo make -j8

RUN pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

