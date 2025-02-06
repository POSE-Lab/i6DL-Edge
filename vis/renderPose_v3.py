from OpenGL.GL import *
import numpy as np
from Shader import Shader
from plyfile import PlyData
from absl import (
    logging,
    flags,
    app)
from vis_pose_wraper import *
from glob import glob
import tqdm
import os

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'md_store_path','',
    'Path the the ply models are stored'

)
flags.DEFINE_string(
    'srv_res_path',"",
    'Service result path, \
    usually found in the folder /root/sanbox/results/XXXXX/image_raw'
)
flags.DEFINE_boolean(
    'vis_inliers_2D',True,
    "Visualize the inliers in 2D"
)
flags.DEFINE_list(
    'res',[640,480],
    'Resolution of input/output image'
)
flags.DEFINE_boolean(
    'vis_inliers_3D',False,
    "Visualize the inliers in 3D"
)
flags.DEFINE_float(
    'op',0.7,
    "Controls the opacity of the overlayed pose"
)
flags.DEFINE_boolean(
    'debug',False,
    'Run in debug mode'
)

def load3DModels(storePath : str) -> list:

    """Loads the provided 3D models of the objects and stores them
    as vertices and face indices.

    Args:
        storePath (str): Path containing the .ply models

    Returns:
        models (list): Vertices and face indices
    """

    models = []
    pbar = tqdm.tqdm(glob(storePath + "/*.ply"),desc="Loading 3D models")
    for md in pbar:

        pbar.set_description(f"Loading {os.path.basename(md)}")
        modelData = PlyData.read(md)
        modelName = os.path.basename(md)

        vertices = np.array(modelData['vertex'][:])
        vertices = vertices.view((vertices.dtype[0], len(vertices.dtype.names)))
        face_ind = map(lambda o: np.array(o) ,
                       np.array(modelData['face'].data['vertex_indices'])) 
        models.append([modelName,vertices.astype(np.float32).flatten(),
                      np.array(list(face_ind)).flatten()])
    return models

def main(arg):

    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    # load the 3D models
    logging.info("Loading 3D models from %s",FLAGS.md_store_path)
    models = load3DModels(FLAGS.md_store_path)

    test_folders = [ f.path for f in os.scandir(FLAGS.srv_res_path) if f.is_dir() ]
    
    # initialize the renderer
    FBO_WIDTH,FBO_HEIGHT = int(FLAGS.res[0]),int(FLAGS.res[1])
    renderer = Renderer(bufferSize=(FBO_WIDTH,FBO_HEIGHT))
    renderer.load_shaders("basic_lighting_vrt.txt", "basic_lighting.txt", None)
    renderer.CreateFramebuffer(GL_RGB32F, GL_RGBA, GL_FLOAT)

    glEnable(GL_DEPTH_TEST)
    

    # far and near planes for clipping
    nearP = 1
    farP = 5000

    pbar = tqdm.tqdm(test_folders,desc="Visualizing")
    for tf in pbar:

        try:
            obj_fld = next(os.walk(tf))[-2][0]
            objID = obj_fld.split('_')[-1][0]
            logging.debug("Rendering model with id %s",objID)
            image_path = os.path.join(tf,obj_fld) + "/image_raw.png"
            pose_file = os.path.join(tf,obj_fld) + "/pose.txt"
            current_model = [_ for _ in models if _[0] == objID+".ply"] #TODO
            renderer.create_buffers(current_model[0][1],current_model[0][2],attrs=[2,3,4])

            glBindFramebuffer(GL_FRAMEBUFFER, renderer.framebuffer.ID)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0., 0., 0., 1.)
            glUseProgram(renderer.shader_programm)
            # reafin pose file
            with open(pose_file,'r') as p:
                        temp = p.readlines()
                        pose_status = temp[0]
                        pose = [k.strip('\n').split(',')[:-1] for k in temp[1:5]]
                        K = temp[5].split(',')[:-1]
                        conf = np.round(float(temp[-1]),2)
            logging.debug(K)
            K = np.array(K).reshape(3,3).astype(float)
            proj = [ 2 * K[0,0] / float(FBO_WIDTH),     -2 * 0 / float(FBO_WIDTH) ,     0,    0,
            0,                       2 * K[1,1] / float(FBO_HEIGHT),      0,   0,
            1 - 2 * (K[0,2] / float(FBO_WIDTH)),      2 * (K[1,2] / float(FBO_HEIGHT)) - 1,   -(farP + nearP) / (farP - nearP),       -1,
            0,                       0,                      2 * farP * nearP / (nearP - farP),       0
            ]
            proj = np.array(proj).reshape(4,4)

            RT = np.array(pose).reshape(4,4).astype(float)
            logging.debug(f"\n|---------------------K---------------------|\n{K}")
            logging.debug(f"\n|---------------------RT--------------------|\n{RT}")
            INV_MATR  = np.array([1.0 , 0.0, 0.0, 0.0,
                0.0,-1.0,0.0,0.0,
                0.0,0.0,-1.0,0.0,
                0.0, 0.0, 0.0, 1.0]).reshape((4,4))
            
            
            RT = np.transpose(RT)
            RT = np.dot(RT,INV_MATR)
            res = np.dot(np.array([RT[0,-1],RT[1,-1],RT[2,-1],1.0]),np.linalg.inv(RT))
            #rot = np.array(rot).reshape(3,3)
            
            model_ = pyrr.Matrix44.identity()
            
            uni_vars = {"objectColor": {"value":[1.0, 0.5, 0.31],"type":'glUniform3fv'}, # 1.0, 0.5, 0.31
                    "lightColor":{"value": [1.0, 1.0, 1.0],"type":'glUniform3fv'},
                    "lightPos":{"value": [res[0], res[1] , res[2]],"type":'glUniform3fv'},
                    "viewPos":{"value": [-RT[0,-1], -RT[1,-1], -RT[2,-1]],"type":'glUniform3fv'},
                    "model":{"value": model_,"type":'glUniformMatrix4fv'},
                    "view":{"value":RT,"type":'glUniformMatrix4fv'},
                    "projection":{"value": proj,"type":'glUniformMatrix4fv'},
                    }
            renderer.setUniformVariables(renderer.shader_programm,uni_vars)

            
            glBindVertexArray(renderer.VAO)
            glDrawElements(GL_TRIANGLES, len(current_model[0][2]), GL_UNSIGNED_INT, None)
            
            renderer.ProjectFramebuffer(renderer.framebuffer, (FBO_WIDTH, FBO_HEIGHT))
            #mask = renderer.CaptureFramebufferScene()
            

            #ProjectFramebuffer(rgb_framebuffer, (SCRN_WIDTH, SCRN_HEIGHT))
            mask = renderer.CaptureFramebufferScene(saveRendered=False)
            glfw.swap_buffers(renderer.window)
            
            background = cv.imread(image_path).astype(float)
            label=""
            # if objID == '1':
            #     label = "Makita DFT"
            # elif objID == '2':
            #     label = "Control panel"
            # elif objID == '3':
            #     label = "Makita DFL"
            # elif objID == '4':
            #     label = "Wcp2"
            # elif objID == "5":
            #     label = "speaker frame"
            # else:
            #     raise ValueError("Unsupported object")
            renderer.makeOverlay(background,mask,objID,conf,pose_status,
                                os.path.join(tf,obj_fld)+"/result.png",FLAGS.op)
        except Exception as e:
            continue
        
if __name__ =="__main__":
    app.run(main)