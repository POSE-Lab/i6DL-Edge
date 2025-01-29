#!/usr/bin/env python3

from __future__ import print_function

import multiprocessing

from odl.srv import ObjectPoseService,ObjectPoseServiceResponse
from odl.msg import ObjectPose
from odl.msg import ObjectId
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import rospy

# import epos inference dependencies
import os
import onnxruntime as ort
import os.path
import time
import numpy as np
import cv2
import pyprogressivex
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import transform
from bop_toolkit_lib import visualization
from epos_lib import common
from epos_lib import config
from epos_lib import corresp
from epos_lib import datagen

#import multiprocessing
#from numba import cuda
import gc
import shutil



# import createTfRecords dependencies
import io
import random
from functools import partial
from PIL import Image as ImagePIL

# import create example list dependencies
import glob
import yaml

# save images
import imageio
import png

# health msg
import threading
from odl.msg import SystemHealth

from absl import app
from absl import flags
from absl import logging

from datetime import datetime

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# FLAGS = flags.FLAGS

# flags.DEFINE_string(
#     'test',"hallo",
#     'Test'
# )

# utility functions
def save_profiling(times,savePath):
    with open(savePath+"/latency_metrics.txt",'w') as f:
        f.write(str(times))
def save_EPOS_pose(path,pos,orie,conf,pose_status,objid, K_mat):
    orie = np.array(orie).reshape((3,3))
    _t = np.vstack((orie,np.array([0.0,0.0,0.0]).reshape((1,3))))
    print(_t)
    _f = np.array(pos).reshape((3,1))
    _f = np.append(_f,[1.0])
    print(_f)
    _f = np.hstack((_t,_f.reshape((4,1))))

    filename = path+"/Obj_"+str(objid)+"/pose.txt"
    print("saving pose at ", filename)
    file = open(filename, 'w')
    file.write(pose_status+"\n")
    np.savetxt(file, _f, newline=',\n',delimiter=',',fmt='%f')
    np.savetxt(file, K_mat, newline=', ',delimiter=',',fmt='%f')
    file.write('\n'+str(conf))
    file.close()
    #f.write(str(orie))

def make_test_dirs(parent_path,timestamp,objID):
# dynamically create test directorys based on the number of service calls
    current_working_dir = parent_path+"/Test_"+str(timestamp)
    if not os.path.exists(current_working_dir):
        try:
            os.makedirs(current_working_dir)
        except OSError as e:
            print(e)

    #also create dirs for ach object
    if not os.path.exists(current_working_dir+"/Obj_"+str(objID)):
        try:
            os.makedirs(current_working_dir+"/Obj_"+str(objID))
        except OSError as e:
            print(e)
    if not os.path.exists(current_working_dir+"/Obj_"+str(objID)+"/corr_"):
        try:
            os.makedirs(current_working_dir+"/Obj_"+str(objID)+"/corr_")
        except OSError as e:
            print(e)
    return current_working_dir

def save_Image(path,img,timestamp,objID):

    print("Saving image at ", path+"/Test_"+str(timestamp)+"/Obj_"+str(objID)+"/image_raw.png")
    imageio.imwrite(path+"/Test_"+str(timestamp)+"/Obj_"+str(objID)+"/image_raw.png", img)

def load_image(img_path):

    image = np.array(ImagePIL.open(img_path)).astype(np.float32)
    rgb_image_input = np.array(image).astype(np.float32)

    return rgb_image_input

def save_correspondences(
        scene_id, im_id, timestamp, obj_id, image_path, K, obj_pred,sort_inds, pred_time,
        infer_name, obj_gt_poses, infer_dir,inliers_indices,inliers,total):

    # Add meta information.
    txt = '# Corr format: u v x y z px_id frag_id conf conf_obj conf_frag\n'
    txt += '{}\n'.format(image_path)
    print(inliers_indices.shape)
    print("INLIERS",inliers)
    txt += 'Number of Inliers :{} ,Number of outliers: {}, Ratio: {}\n'.format(inliers,total- inliers,inliers/total)

    # Add intrinsics.
    for i in range(3):
        txt += '{} {} {}\n'.format(K[i, 0], K[i, 1], K[i, 2])

    # Add ground-truth poses.
    txt += '{}\n'.format(len(obj_gt_poses))
    for pose in obj_gt_poses:
        for i in range(3):
            txt += '{} {} {} {}\n'.format(
                pose['R'][i, 0], pose['R'][i, 1], pose['R'][i, 2], pose['t'][i, 0])

    # Sort the predicted correspondences by confidence.
    px_id = obj_pred['px_id'][sort_inds]
    frag_id = obj_pred['frag_id'][sort_inds]
    coord_2d = obj_pred['coord_2d'][sort_inds]
    coord_3d = obj_pred['coord_3d'][sort_inds]
    conf = obj_pred['conf'][sort_inds]
    conf_obj = obj_pred['conf_obj'][sort_inds]
    conf_frag = obj_pred['conf_frag'][sort_inds]

    # Add the predicted correspondences.
    pred_corr_num = len(coord_2d)
    txt += '{}\n'.format(pred_corr_num)
    for i in range(pred_corr_num):
        txt += '{} {} {} {} {} {} {} {} {} {} {}\n'.format(inliers_indices[i],
            coord_2d[i, 0], coord_2d[i, 1],
            coord_3d[i, 0], coord_3d[i, 1], coord_3d[i, 2],
            px_id[i], frag_id[i], conf[i], conf_obj[i], conf_frag[i])

    # Save the correspondences into a file.
    corr_suffix = infer_name
    if corr_suffix is None:
        corr_suffix = ''
    else:
        corr_suffix = '_' + corr_suffix

    corr_path = os.path.join(
        infer_dir, 'corr{}'.format(corr_suffix),str(timestamp)+"_corr")
    print("TIMES CALLED :",im_id)
    with open(corr_path, 'w') as f:
        f.write(txt)

def process_image(self,K,predTime,image,predictions, im_id, scene_id, output_scale, model_store,
        renderer, task_type,corr_path,timestamp):
    """Estimates object poses from one image.

    Args:
      sess: TensorFlow session.
      samples: Dictionary with input data.
      predictions: Dictionary with predictions.
      im_ind: Index of the current image.
      crop_size: Image crop size (width, height).
      output_scale: Scale of the model output w.r.t. the input (output / input).
      model_store: Store for 3D object models of class ObjectModelStore.
      renderer: Renderer of class bop_renderer.Renderer().
      task_type: 6D object pose estimation task (common.LOCALIZATION or
        common.DETECTION).
      infer_name: Name of the current inference.
      infer_dir: Folder for inference results.
      vis_dir: Folder for visualizations.
    """
    # Dictionary for run times.
    run_times = {}
    run_times['prediction'] = predTime

    K = np.array(K).reshape((3,3))

    gt_poses = None
    # Establish many-to-many 2D-3D correspondences.
    time_start = time.time()
    
    corr = corresp.establish_many_to_many(
        obj_confs=predictions[common.PRED_OBJ_CONF][0],
        frag_confs=predictions[common.PRED_FRAG_CONF][0],
        frag_coords=predictions[common.PRED_FRAG_LOC][0],
        gt_obj_ids=[int(scene_id)],
        model_store=model_store,
        output_scale=output_scale,
        min_obj_conf=self.corr_min_obj_conf,
        min_frag_rel_conf=self.corr_min_frag_rel_conf,
        project_to_surface=self.project_to_surface,
        only_annotated_objs=(task_type == common.LOCALIZATION))
    
    run_times['establish_corr'] = time.time() - time_start

    # PnP-RANSAC to estimate 6D object poses from the correspondences.
    time_start = time.time()
    poses = []
    
    
    for obj_id, obj_corr in corr.items():
        
        # tf.compat.v1.logging.info(
        #   'Image path: {}, obj: {}'.format(samples[common.FLAGS.img_path][0], obj_id))

        # Number of established correspondences.
        num_corrs = obj_corr['coord_2d'].shape[0]

        # Skip the fitting if there are too few correspondences.
        min_required_corrs = 6
        if num_corrs < min_required_corrs:
            continue

        # The correspondences need to be sorted for PROSAC.
        if self.use_prosac:    
           
            sorted_inds = np.argsort(obj_corr['conf'])[::-1]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][sorted_inds]
        
        # Select correspondences with the highest confidence.
        if self.max_correspondences is not None \
                and num_corrs > self.max_correspondences:
            # Sort the correspondences only if they have not been sorted for PROSAC.
            if self.use_prosac:
                keep_inds = np.arange(num_corrs)
            else:
                keep_inds = np.argsort(obj_corr['conf'])[::-1]
            keep_inds = keep_inds[:self.max_correspondences]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][keep_inds]

        

        # Make sure the coordinates are saved continuously in memory.
        coord_2d = np.ascontiguousarray(
            obj_corr['coord_2d'].astype(np.float64))
        coord_3d = np.ascontiguousarray(
            obj_corr['coord_3d'].astype(np.float64))
        
        if self.fitting_method == common.PROGRESSIVE_X:
            # If num_instances == 1, then only GC-RANSAC is applied. If > 1, then
            # Progressive-X is applied and up to num_instances poses are returned.
            # If num_instances == -1, then Progressive-X is applied and all found
            # poses are returned.
            if task_type == common.LOCALIZATION:
                num_instances = 1
            else:
                num_instances = -1

            if self.max_instances_to_fit is not None:
                num_instances = min(num_instances, self.max_instances_to_fit)
            
            pose_ests, inlier_indices, pose_qualities = pyprogressivex.find6DPoses(
                x1y1=coord_2d,
                x2y2z2=coord_3d,
                K=K,
                threshold=self.inlier_thresh,
                neighborhood_ball_radius=self.neighbour_max_dist,
                spatial_coherence_weight=self.spatial_coherence_weight,
                scaling_from_millimeters=self.scaling_from_millimeters,
                max_tanimoto_similarity=self.max_tanimoto_similarity,
                max_iters=self.max_fitting_iterations,
                conf=self.required_progx_confidence,
                proposal_engine_conf=self.required_ransac_confidence,
                min_coverage=self.min_hypothesis_quality,
                min_triangle_area=self.min_triangle_area,
                min_point_number=6,
                max_model_number=num_instances,
                max_model_number_for_optimization=self.max_model_number_for_pearl,
                use_prosac=self.use_prosac,
                log=False)

            # Save the established correspondences (for analysis).
            obj_gt_poses = []
            if gt_poses is not None:
                obj_gt_poses = [x for x in gt_poses if x['obj_id'] == obj_id]
            pred_time = float(np.sum(list(run_times.values())))
            image_path = "test"

            # for the confidense calculation
            sort_inds = np.argsort(obj_corr['conf'])[::-1]
            coord_2d = obj_corr['coord_2d'][sort_inds]
            total = len(coord_2d)
            inliers = np.count_nonzero(inlier_indices == 1)
            pose_confidense = (inliers / total) if total!=0 else 0
            

            save_correspondences(
                scene_id, im_id, timestamp, obj_id, image_path, K, obj_corr,sort_inds, pred_time,
                "", obj_gt_poses, corr_path,inlier_indices,inliers,total)
            
            pose_est_success = pose_ests is not None
            if pose_est_success:
                for i in range(int(pose_ests.shape[0] / 3)):
                    j = i * 3
                    R_est = pose_ests[j:(j + 3), :3]
                    t_est = pose_ests[j:(j + 3), 3].reshape((3, 1))
                    poses.append({
                        'scene_id': scene_id,
                        'im_id': im_id,
                        'obj_id': obj_id,
                        'R': R_est,
                        't': t_est,
                        'score': pose_qualities[i],
                    })

        elif self.fitting_method == common.OPENCV_RANSAC:
            # This integration of OpenCV-RANSAC can estimate pose of only one object
            # instance. Note that in Table 3 of the EPOS CVPR'20 paper, the scores
            # for OpenCV-RANSAC were obtained with integrating cv2.solvePnPRansac
            # in the Progressive-X scheme (as the other methods in that table).
            pose_est_success, r_est, t_est, inliers = cv2.solvePnPRansac(
                objectPoints=coord_3d,
                imagePoints=coord_2d,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=self.max_fitting_iterations,
                reprojectionError=self.inlier_thresh,
                confidence=0.99,  # FLAGS.required_ransac_confidence
                flags=cv2.SOLVEPNP_EPNP)

            if pose_est_success:
                poses.append({
                    'scene_id': scene_id,
                    'im_id': im_id,
                    'obj_id': obj_id,
                    'R': cv2.Rodrigues(r_est)[0],
                    't': t_est,
                    'score': 0.0,  # TODO: Define the score.
                })

        else:
            raise ValueError(
                'Unknown pose fitting method ({}).'.format(self.fitting_method))

    run_times['fitting'] = time.time() - time_start
    run_times['total'] = np.sum(list(run_times.values()))

    # Add the total time to each pose.
    for pose in poses:
        pose['time'] = run_times['total']

    return poses,pose_confidense,run_times
class TRT_engine:
    def __init__(self,engine_path,input_tensor) -> None:
        
        
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        input_shape = self.context.get_tensor_shape(input_tensor)
        input_nbytes = trt.volume(input_shape) *np.dtype(np.float32).itemsize

        self.input_gpu = cuda.mem_alloc(input_nbytes)

        self.stream = cuda.Stream()

        #Allocate output buffer
        self.cpu_outputs=[]
        self.gpu_outputs=[]
        for i in range(1,5):
            self.cpu_outputs.append(cuda.pagelocked_empty(tuple(self.context.get_binding_shape(i)), dtype=np.float32))
            self.gpu_outputs.append(cuda.mem_alloc(self.cpu_outputs[i-1].nbytes))

    def predict(self,image):

        print("Predict with TRT")
        print(threading.current_thread())
        #print(cuda.Device)
        eval_start_time = time.time()

        
        cuda.memcpy_htod_async(self.input_gpu,image,self.stream)
        #Copy inouts
        self.context.execute_async_v2(bindings=[int(self.input_gpu)] + [int(outp) for outp in self.gpu_outputs] , 
            stream_handle=self.stream.handle)

        for i in range(4):
            cuda.memcpy_dtoh_async(self.cpu_outputs[i], self.gpu_outputs[i], self.stream)

        self.stream.synchronize()

        eval_time_elapsed = time.time() - eval_start_time

        predictions={'pred_frag_conf':self.cpu_outputs[2],
                    'pred_frag_loc':self.cpu_outputs[1],
                    'pred_obj_conf':self.cpu_outputs[0],
                    'pred_obj_label':self.cpu_outputs[3]}

        # if FLAGS.verbose:
        #     # print("Array_1",self.cpu_outputs[0])
        #     # print("Array_2",self.cpu_outputs[1])
        #     # print("Array_3",self.cpu_outputs[2])
        #     # print("Array_4",self.cpu_outputs[3])
        #     print(predictions)
        #     print("Inference took : ",eval_time_elapsed)
        
        print(eval_time_elapsed)

        return predictions,eval_time_elapsed
    def visualize(self):
        
        np_image = np.array(self.cpu_outputs[3])
        np_image[np_image !=0]=255
        aa = np.reshape(np_image.astype("uint8"),(180,320))
        image = Image.fromarray(aa)
        image.show()
# Class for actual epos functionality
class eposInfer:
    def __init__(self,attrs):
        
        # automatically make attributes from input dictionary
        for key in attrs:
            setattr(self, key, None if attrs[key] == 'None' else attrs[key])


        self.model_store_frag = datagen.ObjectModelStore(
        dataset_name='felice',
        model_type='cad',
        num_frags=self.num_frags,
        prepare_for_projection=self.project_to_surface) # dataset_name='carObj1'

        # Fragment the 3D object models.
        self.model_store_frag.fragment_models()
        frag_centers = self.model_store_frag.frag_centers
        frag_sizes = self.model_store_frag.frag_sizes

        self.model_store = datagen.ObjectModelStore(
            dataset_name='felice',
            model_type='cad',
            num_frags=self.num_frags,
            frag_centers=frag_centers,
            frag_sizes=frag_sizes,
            prepare_for_projection=self.project_to_surface)
        self.model_store.load_models()

        

        # warmup
        print("Initiating session for inference...")
        
        if self.method == 'onnx':
            # Initial configuration of ONNX and TensorRT runtime
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers= [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1", "cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
            self.sess = ort.InferenceSession(self.onnx,providers=providers,sess_options=sess_options)
        elif self.method == 'trt':
            self.trtEngine = TRT_engine(self.trt,"rgb_image_input:0")
         
        dummy_image = load_image(self.init_image_path)
        #warmup 
        print("Warming up...")
        for _ in range(10):
            if self.method == 'onnx':
                result = self.sess.run(["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],
                {"rgb_image_input:0":np.array(dummy_image).astype(np.float32)})
            elif self.method == 'trt':
                predictions,predTime = self.trtEngine.predict(dummy_image)
        #predictions,predTime = self.trtEngine.predict(dummy_image)
        print("Done warming up")


    def predictPose(self,K,objID,rgb_image_input,method,corr_path,timestamp):

        # rgb_image_input = load_image("/home/lele/Codes/epos/epos_optim/data/test_images/test_primesence/000001/rgb/000008.png")
        rgb_image_input_ = np.array(rgb_image_input).astype(np.float32)
        

        infer_time_start = time.time()

        if method == 'onnx':
            #warmap session - initiolization time
            result = self.sess.run(["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],
            {"rgb_image_input:0":rgb_image_input_})

            predictions={'pred_frag_conf':result[0],
                        'pred_frag_loc':result[1],
                        'pred_obj_conf':result[2],
                        'pred_obj_label':result[3]}
        elif method == 'trt':
            print("Inference with TRT")
            #trt_ = TRT_engine(self.trt,"rgb_image_input:0")
            predictions,predTime = self.trtEngine.predict(rgb_image_input_)

        # if resolution == [640,480]:
        #     # then don't query K from the camera ros topic
        #     # and specify manually cause of the bug in fx,fy values
        #     K = [423.87302656,0.0,318.68035889,0.0,423.07466667,242.97497559,0.0,0.0,1.0]       
        print("K USED ",K)
        infer_time_elapsed = time.time() - infer_time_start
        poses,confidense,runtimes = process_image(
                self,
                K=K,
                predTime = infer_time_elapsed,
                image="",
                predictions= predictions, #TODO: dictiomary
                im_id=0,
                scene_id=objID,
                output_scale=(1.0 / self.decoder_output_stride[0]),
                model_store=self.model_store,
                renderer=None,
                task_type=self.task_type,
                corr_path=corr_path,
                timestamp=timestamp)
        print(runtimes)
        return poses, confidense
# Class for service related data handling
class cameraInput(eposInfer):
    def __init__(self,eposObj):
        #self.times_called = 0
        # initialize to latest possible datetime. Format it as
        # <Year>_<month>_<day>_<hour><minutes><seconds>                                                       
        self.timestamp = datetime.max.strftime("%Y_%m_%d_%H%M%S") 
        self.current_working_dir = ''
        self.eposObj = eposObj
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
        self.pub = rospy.Publisher('/ICCS/ObjectDetectionAndLocalization/ObjectPose', ObjectPose, queue_size=10) # TODO
        self.ImageReady = False

        if self.eposObj.clear_store_path:
            try:
                shutil.rmtree(self.eposObj.vis_path+"/image_raw/*")
            except OSError as e:
                print(e)

    def callbackRGBImage(self, data):

        if self.enableInputCallbacks == True:
            try:
                #self.times_called += 1
                self.timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                time_get_img_from_camera = time.time()
                rgb_raw = CvBridge().imgmsg_to_cv2(data, desired_encoding="bgr8")
                #rgb_raw = np.zeros((480,640,3),dtype=np.float32)
                self.rgbImageTimestamp = data.header.stamp
                #self.ImageReady =True # uncomment this line and comment the rst below until the exception if you want to run with pre recorded. Also comment the part of deleting files3 and 4
                self.rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)

                
                self.rgb = cv2.imread("/root/sandbox/init_images/images_1280_720/000011.png")
                self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
                if self.rgb.size != 0:
                    # if self.rgb.size != self.eposObj.input_resolution:
                    #     raise ValueError(f"Image resolution is configured by the .config to be {self.eposObj.input_resolution} but the cameras stream provided an image \
                    #                      of size {self.rgb.shape}")
                    self.ImageReady = True
                print("**Image to take image from camera: ", str(time.time() - time_get_img_from_camera))
                
                
                if self.saveImage:
                    path = self.eposObj.vis_path +"/image_raw" # TODO
                    print("image was taken at ", self.timestamp)
                    self.current_working_dir = make_test_dirs(path,self.timestamp,self.objectId)
                    
                    if self.saveImage:
                        
                        save_Image(path,self.rgb,self.timestamp,self.objectId)

            except CvBridgeError as e:
                print(e)

    def callbackCameraInfo(self, data):
        if self.enableInputCallbacks == True:
            if self.eposObj.input_resolution == [1280,720]:
                self.K = data.K 
            else : self.K = self.eposObj.K_640  

    def get_pose(self, req):
            
            print("*** Pose request ***")
            profiler = {}
            
            # handle requested parameters
            if not req.objID in self.eposObj.obj_ids:
                raise ValueError(f"Model {req.ObjID} not supported")
            else:
                self.objectId = req.objID

            
            # Flag for capturing image when service is called
            self.enableInputCallbacks = True
            
            time_Object_Pose_s = time.time()
            pose = ObjectPose()
            
            profiler['Object_pose'] = str(time.time() - time_Object_Pose_s)

            waiting_for_image = time.time()
            while self.rgb is None or self.K is None or not self.ImageReady:
                pass
            profiler['wating_for_image'] = str(time.time() - waiting_for_image)

            time_s = time.time()
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
            pred_time_s = time.time()
            corr_save_path = self.eposObj.vis_path +"/image_raw"+"/Test_"+str(self.timestamp)+"/Obj_"+str(self.objectId)
            self.posesPredicted,self.pose_confidense = self.eposObj.predictPose(self.K,
                                                                                    req.objID,
                                                                                    self.rgb,
                                                                                    self.eposObj.method,
                                                                                    corr_save_path,
                                                                                    self.timestamp)
            self.pose_status = 'accepted' if self.pose_confidense >=0.5 else 'rejected'
            profiler['predictPose'] = str(time.time() - pred_time_s)

            

            assigne_time_s = time.time()
            # Assign predicted pose
            first_im_poses_num = len(self.posesPredicted)
            for i in range(first_im_poses_num):
                if self.objectId == self.posesPredicted[i]['obj_id']:
                    sc = self.posesPredicted[i]['score']
                    pos = self.posesPredicted[i]['t']
                    orie = self.posesPredicted[i]['R'].flatten('C')
                #uL = [0, 0]
                #lR = [0, 0]
            
            if self.eposObj.save_Pose:
                save_EPOS_pose(self.current_working_dir,pos,orie,self.pose_confidense,self.pose_status,self.objectId, self.K)
            # create assign values to the results message
            pose.timestamp = rospy.get_rostime()  # .get_rostime()
            pose.timestampImage = timeImage #rospy.Time.now()
            pose.score = self.pose_confidense
            pose.objID = req.objID
            #if self.pose_status == 'accepted':
            pose.position = pos
            pose.orientation = orie
            pose.uLCornerBB = uL
            pose.lRCornerBB = lR

            # clean up inputs to get prepared for the next call
            self.rgb = None
            #self.depth = None
            self.K = None

            profiler['assginTime'] = str(time.time() - assigne_time_s)

            # # cleanup folders
            # files3 = glob.glob('/home/lele/Codes/epos/store/tf_data/example_lists/*.txt', recursive=True)
            # files4 = glob.glob('/home/lele/Codes/epos/store/tf_data/*.tfrecord', recursive=True)
            #
            # for f in files3:
            #    try:
            #        os.remove(f)
            #    except OSError as e:
            #        print("Error: %s : %s" % (f, e.strerror))
            # for f in files4:
            #    try:
            #        os.remove(f)
            #    except OSError as e:
            #        print("Error: %s : %s" % (f, e.strerror))

            publish_time = time.time()
            self.pub.publish(pose)
            profiler['PublishTime'] = str(time.time() - publish_time)

            # clean up cuda
            gc_collect_time = time.time()
            gc.collect() #https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
            profiler['Gc_collect_time'] = str(time.time() - gc_collect_time)

            print("***FINAL TIME: ",profiler)
            print("****END TIME****",str(time.time() - time_s))
            print("The confidense calculated for the pose was ",self.pose_confidense)
            profiler['final_time'] = str(time.time() - time_s)

            save_profiling(profiler,self.eposObj.vis_path +"/image_raw"+"/Test_"+str(self.timestamp)+"/Obj_"+str(self.objectId))
            return ObjectPoseServiceResponse(pose) 
    
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

def runODL():
    
    #print("******* ", FLAGS.test)
    # initilize epos once
    
    script_dir = os.path.dirname(__file__) # absolute path of the directory invoking the script
    conf_filename = "config.yml"
    conf_full_path = os.path.join(script_dir, conf_filename)
    print("Loading configuration from", conf_full_path, "...")
    with open(conf_full_path,'r') as c: 
        data = yaml.load(c, Loader=yaml.SafeLoader)
    print(yaml.dump(data))
    print("********Initializing********")
    infer = eposInfer(data)

    print(vars(infer))

    camInput = cameraInput(infer)
    rospy.init_node('ObjPose', anonymous=True)

    sendHealthMsg()
    #rospy.Subscriber('/device_0/sensor_1/Color_0/image/data', Image, camInput.callbackRGBImage) # run my bags from 202207
    rospy.Subscriber('/camera/color/image_raw', Image, camInput.callbackRGBImage)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camInput.callbackCameraInfo)
    #rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, camInput.callbackDepthImage)
    s = rospy.Service('object_pose', ObjectPoseService, camInput.get_pose)
    rospy.spin() 


if __name__ == "__main__":
    #app.run(runODL)
    runODL()