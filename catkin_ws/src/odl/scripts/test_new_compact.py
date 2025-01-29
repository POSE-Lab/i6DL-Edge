import os
import yaml
import sys
from absl import app,flags,logging

sys.path.append("/root/catkin_ws/src/odl/scripts")
from initialize import EposModel
from profiler import Profiler
from camera import cameraInput
from ToolstandDetector_stable import ToolstandDetector

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import rospy
import threading
from odl.msg import SystemHealth
from odl.srv import ObjectPoseService,ObjectPoseServiceResponse


def publisherHealth():
    
    pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/SystemHealth', SystemHealth, queue_size=10)
    #rospy.init_node('healthPub', anonymous=True)  # define the ros node - publish node
    rate = rospy.Rate(10)  # 10hz frequency at which to publish

    if not rospy.is_shutdown():
        msg = SystemHealth()
        msg.timestamp = rospy.Time.now()  # .get_rostime()
        msg.status = "OK"

        rospy.loginfo(msg)  # to print on the terminal
        pub.publish(msg)  # publish
        rate.sleep()

def sendHealthMsg():
  threading.Timer(5.0, sendHealthMsg).start()
  #status = "OK"
  publisherHealth()
  
def runODL(args):
    
    script_dir = os.path.dirname(__file__) # absolute path of the directory invoking the script
    conf_filename = "config.yml"
    conf_full_path = os.path.join(script_dir, conf_filename)
    
    logging.info("Loading configuration from " + conf_full_path + "...")
    with open(conf_full_path,'r') as c: 
        data = yaml.load(c, Loader=yaml.SafeLoader)
        
    logging.info("******Initializing*******")
    
    # instantiate a profiler object
    profiler = Profiler()
    
    # initlize epos model
    eposObj = EposModel(data,profiler)

    # initialize Toolstand detector
    tD = ToolstandDetector(os.path.join(script_dir,"config_toolstand.yaml"))

    if eposObj.warmup:
        eposObj.warm_up("rgb_image_input:0",10)
    
    # initialize camera class
    if eposObj.testMode:
        print("WARNING: Test mode is activated!")
    camera = cameraInput(eposObj,tD,profiler,testMode=eposObj.testMode)
    
    # initialize /objPose node
    rospy.init_node('ObjPose', anonymous=True)
    
    # asychronously publish health message
    sendHealthMsg()
    
    # subscrive to the camera topics
    rospy.Subscriber('/camera/color/image_raw', Image, camera.callbackRGBImage)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera.callbackCameraInfo)
    
    rospy.Service('object_pose', ObjectPoseService, camera.get_pose)
    rospy.spin()

if __name__ == "__main__":
    app.run(runODL)
