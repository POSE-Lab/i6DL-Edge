from epos_lib import datagen
from TensorRT import TRT_engine
import onnxruntime as ort
import numpy as np
import os
from profiler import Profiler
from inout import load_image
from postprocessing import process_image
from PIL import Image
from utilsHF import loadDecimatedModels



class EposModel:
    def __init__(self,attrs,profiler: Profiler) -> None:
        
        # automatically make attributes from input dictionary
        self.profiler = profiler
        self.profiler.addTrackItem("epos_model_init")
        self.profiler.start("epos_model_init")
        for key in attrs:
            setattr(self, key, None if attrs[key] == 'None' else attrs[key])
        
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        # load models for fragmentation
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

        # load the object 3D points for projection
        print("[Initialize]: Loading decimated models...")
        self.decimatedModels = loadDecimatedModels(self.decimated_objects_path)

        self.profiler.stop("epos_model_init")

    def warm_up(self,input_tensor_name : str,steps : int) -> None:

        self.profiler.addTrackItem(self.method+"_warmup")
        self.profiler.start(self.method+"_warmup")
        
        # initilize the selected method
        if self.method == 'trt':
            
            self.trtEngine = TRT_engine(self.trt,input_tensor_name,self.profiler)

        elif self.method == 'onnx':

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers= [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1", "cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
            self.sess = ort.InferenceSession(self.onnx,providers=providers,sess_options=sess_options)

        # load the dummy image 
        dummy_image = load_image(self.init_image_path)

        print("Warming up...")

        for _ in range(steps):

            if self.method == 'onnx':
                result = self.sess.run(["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],
                {input_tensor_name:np.array(dummy_image).astype(np.float32)})
            elif self.method == 'trt':
                predictions,predTime = self.trtEngine.predict(dummy_image)
    
        self.profiler.stop(self.method+"_warmup")
        print("Done warming up")

    def predictPose(self,K,objID,rgb_image_input,corr_path,timestamp):

        rgb_image_input_ = np.array(rgb_image_input).astype(np.float32)
        

        self.profiler.addTrackItem("predict_pose")
        self.profiler.start("predict_pose")

        if self.method == 'onnx':
            #warmap session - initiolization time
            result = self.sess.run(["Softmax:0","Reshape_1:0","Softmax_1:0","ArgMax:0"],
            {"rgb_image_input:0":rgb_image_input_})

            predictions={'pred_frag_conf':result[0],
                        'pred_frag_loc':result[1],
                        'pred_obj_conf':result[2],
                        'pred_obj_label':result[3]}
            
        elif self.method == 'trt':
            print("Inference with TRT")
            #trt_ = TRT_engine(self.trt,"rgb_image_input:0")
            predictions,predTime = self.trtEngine.predict(rgb_image_input_)

        infer_time_elapsed = self.profiler.stop("predict_pose")

        if objID in [301,310]:
            points_dec_,faces_3d_obj = self.decimatedModels[self.decimatedModels[:,0] == 3,1][0]
        else:
            points_dec_,faces_3d_obj= self.decimatedModels[self.decimatedModels[:,0] == objID,1][0]
        poses,confidense,runtimes = process_image(
                self,
                K=K,
                predTime = infer_time_elapsed,
                image_size=(640,480),
                predictions= predictions, #TODO: dictiomary
                im_id=0,
                scene_id=objID,
                output_scale=(1.0 / self.decoder_output_stride[0]),
                model_store=self.model_store,
                renderer=None,
                task_type=self.task_type,
                corr_path=corr_path,
                timestamp=timestamp,
                profiler=self.profiler,
                obj_3d_points=points_dec_,
                faces_3d_obj=faces_3d_obj)
        
        return poses, confidense,runtimes