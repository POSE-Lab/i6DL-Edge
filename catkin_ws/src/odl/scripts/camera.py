from datetime import datetime
import shutil
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from profiler import Profiler
from inout import *
import time
import sys

from initialize import EposModel
from ToolstandDetector_stable import ToolstandDetector

from odl.srv import ObjectPoseService,ObjectPoseServiceResponse
from odl.msg import ObjectPose
from odl.msg import ObjectId

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import gc
class cameraInput:
    def __init__(self,eposObj: EposModel, tsDetector: ToolstandDetector, profiler: Profiler,testMode=False):
        #self.times_called = 0
        # initialize to latest possible datetime. Format it as
        # <Year>_<month>_<day>_<hour><minutes><seconds>     

        self.testMode = testMode                                                 
        self.timestamp = datetime.max.strftime("%Y_%m_%d_%H%M%S") 
        self.profiler = profiler
        self.current_working_dir = ''
        self.eposObj = eposObj
        self.tsDetector = tsDetector
        self.saveImage = eposObj.save_Image
        self.enableInputCallbacks = False
        self.rgb = None
        #self.depth = None
        self.K = None
        self.rgbImageTimestamp = 0
        self.objectId=0
        self.posesPredicted=None
        self.img1Count=-1
        self.img2Count=-1
        self.started = 0
        self.ended = 0
        self.pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/ObjectPose', ObjectPose, queue_size=10) # TODO
        self.ImageReady = False

        if self.eposObj.clear_store_path:
            try:
                shutil.rmtree(self.eposObj.vis_path+"/image_raw/*")
            except OSError as e:
                print('trying to clear results path, original error:',e)

    def callbackRGBImage(self, data):

        self.started = time.time()
        if self.enableInputCallbacks:
            try:
                #self.times_called += 1
                self.timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                self.profiler.addTrackItem("time_get_img_from_camera") 
                self.profiler.start("time_get_img_from_camera")
                
                if self.testMode:
                    # read image from disc
                    self.rgb = cv2.imread(self.eposObj.test_image_path)
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
                    self.ImageReady = True
                self.profiler.stop("time_get_img_from_camera")
                
                
                if self.saveImage:
                    path = self.eposObj.vis_path +"/image_raw" # TODO
                    print("image was taken at ", self.timestamp)
                    print("Creating direcotories...")
                    self.current_working_dir = make_test_dirs(path,self.timestamp,self.objectId,self.profiler)
                    save_Image(path,self.rgb,self.timestamp,self.objectId,self.profiler)
                
                #print(f"SAVING : {timing_save}")
            except CvBridgeError as e:
                print(e)

    def callbackCameraInfo(self, data):
        if self.enableInputCallbacks == True:
            if tuple(self.eposObj.input_resolution) == (720,1280,3):
                if not self.testMode:
                    self.K = data.K
                else:
                    self.K = [634.364, 0.0, 637.801, 0.0, 633.635, 364.958, 0.0, 0.0, 1.0] 
            else : self.K = data.K#self.eposObj.K_640  

    def get_pose(self, req):
            
        """
        Handles the request and based on the object's ID performs one of the above things:

        - Object 6D pose estimation for a single instance
        - Object 6D pose estimation when multiple instances are specified via the requested ID.
        (i.e. if obj_id == 201,210 run with 2 instances and return the right one and the 
        left one respectively)
        - Toolstand pose estimation via ArUco board detection. To request a toolstand's
        pose (of a certain workstation) with respect to the camera the requested id is:

                    i.e.    obj_id = - (workstation number)
        We descriminate these 3D cases that lead to different actions.
        Raises:
            ValueError: _description_

        Returns:
            _type_: ObjectPoseServiceResponse as described in the Data Model
        """
        
        print(f"Service invoked at {rospy.get_time()}")

        # Flag for capturing image when service is called
        self.enableInputCallbacks = True

        print("*** Pose request ***")
        self.profiler.addTrackItem('camera::get_pose')
        self.profiler.start('camera::get_pose')
        profiler = {}
        
        # handle requested parameters
        if req.objID > 0:
            if not str(req.objID) in self.eposObj.object_map:
                self.enableInputCallbacks = False
                raise ValueError(f"Model {req.objID} not supported")
            else:
                self.objectId = int(self.eposObj.object_map[str(req.objID)])
                print("Looking for object: ",self.objectId)
        else:
            if not str(req.objID) in self.tsDetector.toolstand_map:
                self.enableInputCallbacks = False
                raise ValueError(f"Toolstand of workstation {-req.objID} not supported")
        
        
        
        self.profiler.addTrackItem("time_Object_Pose_s")
        self.profiler.start("time_Object_Pose_s")
        pose = ObjectPose()
        
        self.profiler.stop("time_Object_Pose_s")

        self.profiler.addTrackItem("waiting_for_image")
        self.profiler.start("waiting_for_image")
        
        
        while self.rgb is None or self.K is None or not self.ImageReady:
            pass
        self.profiler.stop("waiting_for_image")

        #print("---------------Now image is ready and i can continue with pose estimation", self.ImageReady)

        # Flag for verifying image writing is complete
        self.enableInputCallbacks = False
        self.ImageReady = False # to be able to run multiple times without unexpected empty data

        # Initializations
        rgb = self.rgb
        #depth = self.depth
        # fx = 640.9191284179688 #* im_scale
        # fy = 639.4614868164062 #* im_scale
        # cx = 631.1490478515625 #* im_scale
        # cy = 363.1187744140625 #* im_scale

        # K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        timeImage=self.rgbImageTimestamp

        sc = 0.0
        pos = [0.0, 0.0, 0.0]
        orie = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        uL = [0, 0]
        lR = [0, 0]

        # Run epos prediction
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #print("*****K*****",self.K)
        if req.objID > 0:

            self.profiler.addTrackItem("pred_time_s")
            self.profiler.start("pred_time_s")
            corr_save_path = self.eposObj.vis_path +"/image_raw"+"/Test_"+str(self.timestamp)+"/Obj_"+str(self.objectId)

            # handle wheter the is multiple istances of an object based on the id given
            self.profiler.addTrackItem("TOTAL_PREDICT_POSE:any_method")
            self.profiler.start("TOTAL_PREDICT_POSE:any_method")
            self.posesPredicted,self.pose_confidense,self.runtimes = self.eposObj.predictPose(self.K,
                                                                                    self.objectId,
                                                                                    self.rgb,
                                                                                    corr_save_path,
                                                                                    self.timestamp)
         

            # assign the predicted pose
            if req.objID in [201,210]:
                # return the the left or the right instance
                keep_instance = "left" if str(req.objID).find('1')==1 else "right"
                if keep_instance == "left":
                    prd_pose = self.posesPredicted[0] if self.posesPredicted[0]['t'][0] < self.posesPredicted[1]['t'][0] else self.posesPredicted[1]
                else:
                    prd_pose = self.posesPredicted[0] if self.posesPredicted[0]['t'][0] > self.posesPredicted[1]['t'][0] else self.posesPredicted[1]
                
                if len(self.posesPredicted) == 1:
                    sc = -1.0
                else:
                    sc = prd_pose['score']
                pos = prd_pose['t']
                orie = prd_pose['R'].flatten('C')
                uL = prd_pose['UL']
                lR = prd_pose['LR']

            else:
                first_im_poses_num = len(self.posesPredicted)
                for i in range(first_im_poses_num):
                    if self.objectId == self.posesPredicted[i]['obj_id']:
                        sc = self.posesPredicted[i]['score']
                        pos = self.posesPredicted[i]['t']
                        orie = self.posesPredicted[i]['R'].flatten('C')
                        uL = self.posesPredicted[i]['UL']
                        lR = self.posesPredicted[i]['LR']
                    #uL = [0, 0]
                    #lR = [0, 0]
            
            
            self.profiler.stop("TOTAL_PREDICT_POSE:any_method")
            self.pose_status = 'accepted' if self.pose_confidense >=0.5 else 'rejected'
            self.profiler.stop("pred_time_s")

            
           
        else:

            self.tsDetector.K = self.K
            td_pose,score = self.tsDetector.detect(self.rgb,str(req.objID))
            self.pose_confidense = score
            self.pose_status = 'accepted'
            # constract the ouptut message
            sc = score
            pos = td_pose[:-1,-1]
            orie = td_pose[:3,:3].flatten('C')
            uL = [0,0]
            lR = [0,0]

            if self.tsDetector.vis:
                self.tsDetector.visualize("/root/sandbox/toolstand_test/")
            # detect the toolstand
        # if num_instances == 2:

        #     # return the pose by the keep instance
        #     poses = poses[0] if poses[0]['t'][0] < poses[1]['t'][0] else poses[1]['t'][0]
        #     print(poses)
        # Assign predicted pose

        
        if self.eposObj.save_Pose:
            save_EPOS_pose(self.current_working_dir,pos,orie,self.pose_confidense,self.pose_status,self.objectId, self.K)
        # create assign values to the results message
        pose.timestamp = rospy.get_rostime()  # .get_rostime()
        pose.timestampImage = timeImage #rospy.Time.now()
        pose.score = sc
        pose.objID = req.objID
        #if self.pose_status == 'accepted':
        pose.position = pos
        pose.orientation = orie
        #print([self.bb[0][0],self.bb[0][1]])
        pose.uLCornerBB = list(uL)
        pose.lRCornerBB = list(lR)

        # clean up inputs to get prepared for the next call
        self.rgb = None
        #self.depth = None
        self.K = None

        self.profiler.addTrackItem("publish_time")
        self.profiler.start("publish_time")
        
        self.pub.publish(pose)
        self.profiler.stop("publish_time")

        # clean up cuda
        self.profiler.addTrackItem("gc_collect_time")
        self.profiler.start("gc_collect_time")
        gc.collect() #https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
        self.profiler.stop("gc_collect_time")

        self.profiler.stop('camera::get_pose')

        save_profiling(self.profiler,self.eposObj.vis_path +"/image_raw"+"/Test_"+str(self.timestamp)+"/Obj_"+str(self.objectId)+"/profiling.json")
        self.ended = time.time() - self.started
        print(f"Service reponded at : {rospy.Time.now()}")
        return ObjectPoseServiceResponse(pose)  