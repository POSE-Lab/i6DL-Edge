if [ $# -lt 2 ]; then
  echo -e "At least 2 arguments should be provided.\nMandatory: a valid docker image name, image tag\noptional: environment parameters file"
  exit 1
fi

IMAGE=$1
TAG=$2

USER_CONTAINER=root
WS_CONTAINER="/$USER_CONTAINER/catkin_ws/"
WS_HOST="$(pwd)/../catkin_ws"
SANDBOX_HOST="$(pwd)/../sandbox/"
SANDBOX_CONTAINER="/$USER_CONTAINER/sandbox/"

xhost +

if [ -z "$3" ]; then
echo "no environment file chosen"
  nvidia-docker run -it --privileged \
  -v "$SANDBOX_HOST:$SANDBOX_CONTAINER:rw" \
  -v "$WS_HOST:$WS_CONTAINER:rw" \
  -v "$HOME/.bashrc:/$USER_CONTAINER/.bashrc:rw" \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
    --init  \
      --env="NVIDIA_DRIVER_CAPABILITIES=all" \
      --env="DISPLAY=$DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
      ${IMAGE}:${TAG} \
      bash

else 
  ENV_FILE=$3
  nvidia-docker run -it --privileged \
  -v "$SANDBOX_HOST:$SANDBOX_CONTAINER:rw" \
  -v "$WS_HOST:$WS_CONTAINER:rw" \
  -v "$HOME/.bashrc:/$USER_CONTAINER/.bashrc:rw" \
  --env-file $ENV_FILE \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
    --init  \
      --env="NVIDIA_DRIVER_CAPABILITIES=all" \
      --env="DISPLAY=$DISPLAY" \
      --env="QT_X11_NO_MITSHM=1" \
      --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
      ${IMAGE}:${TAG} \
      bash
fi 
