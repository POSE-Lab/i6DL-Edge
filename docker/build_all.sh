# utility script for building the Docker images
if [ $# -ne 2 ]; then
  echo -e "Please provide a valid image tag and target architecture"
  exit 1
fi
TAG=$1
ARCH=$2 
#DOCKER_ID=$1
docker build -t base-$ARCH:$TAG -f ./$ARCH/base-$ARCH/Dockerfile .. && \
docker build -t custom-ros-$ARCH:$TAG --build-arg="TAG=$TAG" -f ./$ARCH/custom-ros-$ARCH/Dockerfile .. && \
docker build -t realsense-$ARCH:$TAG --build-arg="TAG=$TAG" -f ./$ARCH/realsense-$ARCH/Dockerfile .. && \
docker build -t epos-$ARCH:$TAG --build-arg="TAG=$TAG" -f ./$ARCH/epos-$ARCH/Dockerfile .. && \
docker build -t main-$ARCH:$TAG --build-arg="TAG=$TAG" -f ./$ARCH/main-$ARCH/Dockerfile ..