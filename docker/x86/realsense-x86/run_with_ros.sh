nvidia-docker run -it --privileged \
--net=host \
--env="DISPLAY" \
  --init  \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    main-no-ros \
    bash
