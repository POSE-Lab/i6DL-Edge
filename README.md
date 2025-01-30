# i6DL-Edge
The module uses as baseline method the [EPOS](https://github.com/thodan/epos) (Estimating 6D Pose of Objects with Symmetries) method, adapted as a ROS1 service for integration on robotic platforms. Moreover, optimizations for improved time performance have been integrated, as well as support for deployment in Docker containers. The module can run on **x86_64** and **aarch64/ARM** architectures using Docker containers, in two modes:

- Online mode: the module receives input from an Intel RealSense camera and estimates the 6D pose for the object of interest using the camera's intrinsics.

- Test mode: the module reads an image from disk and estimates the 6D pose using pre-determined intrinsics.    

Inference is supported

- for the ONNX inference engine, using the trained models we provide (see section [Data](#data))

- for the TensorRT inference engine (see TensorRT inference)

## <a name="data"></a> Data
You can use the [IndustryShapes dataset](https://zenodo.org/records/14616197) to test the module with our [pre-trained models](https://ntuagr-my.sharepoint.com/:f:/r/personal/psapoutzoglou_ntua_gr/Documents/FELICE/DATA_CODE_MODELS/trained_models?csf=1&web=1&e=Wda6vx).

## Test environment
- x86: Ubuntu 20.04, [ROS Noetic](http://wiki.ros.org/noetic), [Intel RealSense D455](https://www.intelrealsense.com/depth-camera-d455/)
- aarch64: Linux 5.10.104-tegra (equivalent of Ubuntu 20.04 for NVIDIA Jetson Orin/AGX platforms), ROS Noetic, Intel RealSense D455
    

# 1. Instructions
## 1.1. Setup

Steps that should be carried out before launching the service. "Host" refers to the (physical or virtual) machine on which the Docker container will be deployed. "Root" directory refers to the repo directory (i.e. /some/path/i6DL-Edge)

1. Clone the repo: `git clone --recursive https://github.com/POSE-Lab/i6DL-Edge.git`

2. If you plan on running the rosmaster on the same host as the module, you also need to install ROS.

2. Add the following lines to `.bashrc` in the host

```
export ROS_MASTER_URI=http://<IP IN WHICH ROSMASTER/ROSCORE RUNS>:<PORT>/

export ROS_IP=<IP OF HOST>

export ROS_HOSTNAME=<IP OF HOST>
```

3. Run `source ~/.bashrc` in every terminal so that the changes will be used.

4. In the root directory create a folder named `sandbox` which will hold all necessary data (images, models, results), with the following subfolders:
    - models_cad: PLY 3D models of the objects.
    - decimated_models_prj: folder with decimated 3D models
    - tf_models: folder to keep trained models.
    - init_images: Folder to include images to be used in the initialization/warmup phase when running the module. Images should be of dimensions either 1280 x 720 or 640 x 480.
    - (optional) test_images: Images that can be used for test mode.
Populate the models_cad, tf_models, init_images, test_images folders using the IndustryShapes dataset.

5. Change to `docker` directory and build the Docker images 
```
./build_all.sh <image_tag> <target_architecture>
```
where `<target_architecture>` = `x86` or `arm`

1.2. Running the service

    Edit config.yml in catkin_ws/src/odl/scripts appropriately
    From the base directory, run the init_session.sh script with arguments: session name, docker image, docker image tag, camera resolution. E.g. ./init_session.sh test_session main-arm latest 640 This will create a tmux session with 4 panes. Pane 0: executes launch_camera.sh Pane 1: executes entrypoint.sh Pane 2: executes run_new.sh
    If you want to request a pose for testing purposes, execute run_new.sh in pane 3, type rosservice call /object_pose followed by tab x 2 and input the object ID.
    The image taken from the camera and the estimated pose are saved as image_raw.png and pose.txt respectively under /root/sandbox/results/image_raw/Test_<timestamp>/<Object ID>, where timestamp is the date and time the service is called

4. Misc info & troubleshooting
Killing the tmux session: Ctrl + B, type ":", then enter `kill-session` in the yellow/orange prompt
Detach from tmux session: Ctrl + B, then press d
Cleanly killing all tmux sessions: Run `tmux kill-server`
Changes: Any changes done in catkin_ws, the sandbox, or the .bashrc from either the host or the container are visible from both the host and the container and persist when the container stops/is killed.

    Changes in the odl .py files only require re-running the server (either run ./entrypoint.sh or just rosrun --prefix "/usr/bin/env python3" odl test_new_compact.py)
    Changes in config.yml require re-running the server
    Adding a new ROS package or changing the ROS message definition requires building the catkin workspace again with catkin_make. Please build the workspace inside the container.
    Changes in either 1) any Dockerfile 2) the files in the odl folder (apart from the configuration files) 3) prepare_ws.sh 4) entrypoint.sh require running build_all.sh again, i.e. re-building the docker images

