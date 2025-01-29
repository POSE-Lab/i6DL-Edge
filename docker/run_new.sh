# script to run a new session attached to the most recent running container.
# Takes container id as argument. 

# the ID of the latest running container is on the row 2, 
# column 1 of the 'docker ps -a' output 
docker ps -a
CONTAINER_ID=$(docker ps -a | sed -n '2p' | awk '{print $1}')
echo "Launching new instance of container with ID = $CONTAINER_ID"
xhost +
nvidia-docker exec -it $CONTAINER_ID bash
