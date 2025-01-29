import os
import yaml
import sys
from absl import app,flags,logging

sys.path.append("/root/catkin_ws/src/odl/scripts")
from initialize import EposModel
from profiler import Profiler
from camera import cameraInput

from cv_bridge import CvBridge, CvBridgeError
import cv2

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import rospy
import threading

from odl.msg import ObjectPose
from odl.msg import ObjectId

IMAGE_READY = False
K_READY = False

class camerNS(cameraInput):
    def __init__(self, eposObj, profiler: Profiler, testMode=False):
        super().__init__(eposObj, profiler, testMode)

    def callbackRGBImage(self, data):

        global IMAGE_READY
        global K_READY

        pose = ObjectPose()

        try:
            
            if self.testMode:
                    # read image from disc
                    self.rgb = cv2.imread("/root/sandbox/test_images/image_raw.png")
                    self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
                    #self.rgb = np.zeros(self.eposObj.input_resolution)
            else:
                # get the image from the camera stream
                rgb_raw = CvBridge().imgmsg_to_cv2(data, desired_encoding="bgr8")
                self.rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
            
            self.rgbImageTimestamp = data.header.stamp
            #self.ImageReady =True # uncomment this line and comment the rst below until the exception if you want to run with pre recorded. Also comment the part of deleting files3 and 4
            

            if self.rgb.size != 0:
                if self.rgb.shape != tuple(self.eposObj.input_resolution):
                    raise ValueError(f"Image resolution is configured by the .config to be {self.eposObj.input_resolution} but the cameras stream provided an image \
                                        of size {self.rgb.shape}")
                IMAGE_READY = True

            if IMAGE_READY and K_READY:

                sc = 0.0
                pos = [0.0, 0.0, 0.0]
                orie = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                uL = [0, 0]
                lR = [0, 0]

                print("i am here")
                self.posesPredicted,self.pose_confidense = self.eposObj.predictPose(self.K,
                                                                                    1,
                                                                                    self.rgb,
                                                                                    "",
                                                                                    self.rgbImageTimestamp)
                first_im_poses_num = len(self.posesPredicted)
                for i in range(first_im_poses_num):
                    if 1 == self.posesPredicted[i]['obj_id']:
                        sc = self.posesPredicted[i]['score']
                        pos = self.posesPredicted[i]['t']
                        orie = self.posesPredicted[i]['R'].flatten('C')
                print(orie,pos)
                pose.timestamp = rospy.get_rostime()  # .get_rostime()
                pose.timestampImage = self.rgbImageTimestamp #rospy.Time.now()
                pose.score = self.pose_confidense
                pose.objID = 1
                #if self.pose_status == 'accepted':
                pose.position = pos
                pose.orientation = orie
                pose.uLCornerBB = uL
                pose.lRCornerBB = lR


                self.pub.publish(pose)  

                IMAGE_READY = False
                K_READY = False
                
            
        except CvBridgeError as e:
            print(e)

    def callbackCameraInfo(self, data):
        global K_READY
        if tuple(self.eposObj.input_resolution) == (720,1280,3):
            self.K = data.K 
            K_READY = True
        else : self.K = self.eposObj.K_640  
  
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
    
    # initlize epos modelfrom odl.msg import ObjectPose
    eposObj = EposModel(data,profiler)

    if eposObj.warmup:
        eposObj.warm_up("rgb_image_input:0",10)
    
    # initialize camera class
    camera = camerNS(eposObj,profiler,testMode=False)

    # asychronously publish health message
    rospy.init_node('ObjPose', anonymous=True)
    # subscrive to the camera topics
    rospy.Subscriber('/camera/color/image_raw', Image, camera.callbackRGBImage)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera.callbackCameraInfo)
    

    rospy.spin()
if __name__ == "__main__":
    app.run(runODL)