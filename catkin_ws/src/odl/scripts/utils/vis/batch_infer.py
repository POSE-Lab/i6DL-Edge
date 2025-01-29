from absl import flags,app,logging
from glob import glob
import sys
import yaml
sys.path.append("../../")
from inout import load_image
from initialize import EposModel
from profiler import Profiler
import json
import numpy as np
import os


import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'images','',
    "Folder containing the images to perform inference"
)
flags.DEFINE_string(
    'config','./config.yaml',
    'Configuration file to load'
)
flags.DEFINE_integer(
    'objID',1,
    "Object id to localize in the image. Currently the models\
    are trained for id's 1 and 2 (screwdriver and carDoorPanel)"
)
flags.DEFINE_boolean(
    'vis',False,
    'Visualize predicted poses'
)

def main(args):

    # get the input images in a list
    image_files = glob(FLAGS.images+"/*.png")
    logging.info("Detected %d images",len(image_files))

    logging.info("Loading images...")
    images = [load_image(_) for _ in tqdm.tqdm(image_files,desc="Loading...")]
    logging.info("Done loading images   .")

    # create a profiler object
    profiler = Profiler()

    # load config.yaml
    logging.info("Loading configuration from %s",FLAGS.config)

    with open(FLAGS.config,'r') as c: 
        data = yaml.load(c, Loader=yaml.SafeLoader)
    epos_model = EposModel(data,profiler)

    logging.info("Done loading configuration.")


    if epos_model.warmup:

        logging.info("Warming up...")
        epos_model.warm_up("rgb_image_input:0",10)
        logging.info("Done warming up.")
    
    # get the raw predictions, THIS PARTH OF THE CODE UTILIZES THE GPU
    logging.info("Starting inference on images...")
    predictions = [epos_model.predictPose(epos_model.K_640,FLAGS.objID,o,epos_model.corr_path,"") \
                   for o in tqdm.tqdm(images,desc="Prediction + post processing...")]
    logging.info("Done infrence")
    
    
    logging.info("Writing results to file")

    poses = [np.append(np.array(i[0][0]['R']),np.array(i[0][0]['t'])) for i in predictions]
    confs = [np.array(i[1]) for i in predictions]

    with open("/root/sandbox/results/service_compact_tests/batch_infer_results.txt",'w') as f:
        for i in tqdm.tqdm(range(len(poses))):
            f.write(os.path.basename(image_files[i])+" ")
            np.savetxt(f,np.append(poses[i],confs[i]).reshape(1,-1),newline='\n',delimiter=' ',fmt='%f')

    # visualize
    if FLAGS.vis:
        
        from visualizer import visualizer,PerspectiveCamera
        _config = {
            'modelPath': epos_model.model_3d,
            'shaders': {
                'vert': epos_model.vertex_shader,
                'frag': epos_model.fragment_shader,
                'geom': epos_model.geometry_shader,
            }
        }
        
        _visualizer = visualizer(_config,(640,480))
        print("GOT HERE")
        _visualizer.glConfig()

        PerspectiveCamera_ = PerspectiveCamera(640,480)
        proj = PerspectiveCamera_.fromIntrinsics(epos_model.K[0],epos_model.K[4],epos_model.K[2],epos_model.K[5],1,5000)

        model_ = np.eye(4)
        RT = np.eye(4)
        
        
        uni_vars = {"objectColor": {"value":[1.0, 0.5, 0.31],"type":'glUniform3fv'}, # 1.0, 0.5, 0.31
                    "lightColor":{"value": [1.0, 1.0, 1.0],"type":'glUniform3fv'},
                    "lightPos":{"value": [0.0, 0.0 , 0.0],"type":'glUniform3fv'},
                    "viewPos":{"value": [0.0, 0.0, 0.0],"type":'glUniform3fv'},
                    "model":{"value": model_,"type":'glUniformMatrix4fv'},
                    "view":{"value":RT,"type":'glUniformMatrix4fv'},
                    "projection":{"value": proj,"type":'glUniformMatrix4fv'},
                    }
        
        INV_MATR  = np.array([1.0 , 0.0, 0.0, 0.0,
            0.0,-1.0,0.0,0.0,
            0.0,0.0,-1.0,0.0,
            0.0, 0.0, 0.0, 1.0]).reshape((4,4))
        
        cn=0
        for i in tqdm.tqdm(range(len(poses))):

            _visualizer.glConfig()

            RT = np.eye(4)
            RT[:3,:3] = poses[i][:9].reshape(3,3)
            RT[:3,-1] = poses[i][9:].reshape(3,)
            #print(RT)
            RT = np.transpose(RT)
            RT = np.dot(RT,INV_MATR)
            #print(RT)
            res = np.dot(np.array([RT[0,-1],RT[1,-1],RT[2,-1],1.0]),np.linalg.inv(RT))
            uni_vars["view"]["value"] = RT
            uni_vars["lightPos"]["value"] = [res[0],res[1],res[2]]
            uni_vars["viewPos"]["value"] = [-RT[0,-1], -RT[1,-1], -RT[2,-1]]
            
            _visualizer.setUniformVariables(_visualizer.shader_programm,uni_vars)
            mask = _visualizer.DrawProjectCapture((640,480),epos_model.vis_path+"/result.png")
            _visualizer.overlay(image_files[i],
                                mask,
                                "Screwdriver",
                                conf=np.round(confs[i],2),
                                pose_status="accepted" if confs[i]>0.6 else 'rejected',
                                savePath=epos_model.vis_path + os.path.basename(image_files[i]),
                                buildMask=False,
                                maskFolder=None)

if __name__ == "__main__":
    app.run(main)