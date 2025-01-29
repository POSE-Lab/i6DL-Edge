#!/bin/bash
launch_camera_func() {
    if [[ $1 -eq 640 ]]; then
        echo "Launching ROS node for 640 x 480 resolution"
        roslaunch realsense2_camera rs_camera.launch align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30 
    elif [[ $1 -eq 1280 ]]; then
        echo "Launching ROS node for 1280 x 720 resolution"
        roslaunch realsense2_camera rs_camera.launch enable_accel:=true enable_gyro:=true align_depth:=true color_width:=1280 color_height:=720 color_fps:=30 depth_width:=1280 depth_height:=720 depth_fps:=30
    else
        echo "Wrong resolution"
        exit
    fi
}

if [ "$#" -eq 0 ]; then
    echo "No arguments provided. Please input 1280 for a resolution of 1280 x 720, 640 for a resolution of 640 x 480"
    exit
fi

ROS_PID=$(pidof -x "roslaunch")
echo $ROS_PID
RESOLUTION=$1
STOP_NODE="false"

# handle edge case of multiple roslaunch processes running
# which haven't been killed yet
IFS=', ' read -r -a PID_ARRAY <<< "$ROS_PID"
if (( ${#PID_ARRAY[@]} > 1)); then
    for ID in "${PID_ARRAY[@]}"
    do
        echo "Killing process $ID"
        kill -9 $ID
    done
    echo "Cleaning up..."
    sleep 20
fi

ROS_PID=$(pidof -x "roslaunch")

if [[ $ROS_PID -eq 0 ]] ; then
    echo "Realsense ROS node is not running"
    launch_camera_func $RESOLUTION
else 
    echo "Realsense ROS node is running (PID = $ROS_PID)"
    PROC_DETAILS=$(ps -ax | grep roslaunch)
    if [[ "$PROC_DETAILS" == *"color_width:=640"* ]]; then
        echo "ROS node is launched for resolution: 640 x 480"
        if [[ $RESOLUTION -eq 1280 ]]; then
            STOP_NODE="true"
        fi
    else 
        echo "ROS node is launched for resolution: 1280 x 720"
        if [[ $RESOLUTION -eq 640 ]]; then
            STOP_NODE="true"
        fi
    fi

    if [[ $STOP_NODE == "true" ]]; then
        echo "Killing process $ROS_PID"
        kill -9 $ROS_PID
        sleep 20
        launch_camera_func $RESOLUTION
    fi
fi

