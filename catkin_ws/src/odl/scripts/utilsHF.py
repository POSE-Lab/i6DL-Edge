"""
Utilities
"""
import yaml
from absl import logging
import glob
import sys
sys.path.append("../")
from inout import load_image
import numpy as np
import cv2 as cv
import json
import time
import os
from plyfile import PlyData
from dataclasses import dataclass


@dataclass
class point3D:
    x: float
    y: float
    z: float
@dataclass
class triangle3D:
    v1: point3D
    v2: point3D
    v3: point3D
    
def rotate3D_X(angle_deg):
    
    angle_deg = np.deg2rad(angle_deg)
    R_X = np.array([[1,0,0],
                    [0,np.cos(angle_deg),-np.sin(angle_deg)],
                    [0,np.sin(angle_deg),np.cos(angle_deg)]]) 
    return R_X

def rotate3D_Y(angle_deg):
    
    angle_deg = np.deg2rad(angle_deg)
    R_Y = np.array([[np.cos(angle_deg),0,np.sin(angle_deg)],
                    [0,1,0],
                    [-np.sin(angle_deg),0,np.cos(angle_deg)]])
    return R_Y

def load_config(filename):
    """Loads a configuration file.

    Args:
        filename (str): Path to configuration file

    Returns:
        dict: Loaded yaml dictionary
    """
    with open(filename,'r') as c: 
        data = yaml.load(c, Loader=yaml.SafeLoader)

    return data

def readImagesFromDir(dir):
    """Loads images from "dir" path to disc and
    returns both the image filenames and the loaded images.

    Args:
        dir (str): Path to images

    Returns:
        tuple: Images list (paths) , Images loaded to disk (arrays)
    """
    images = glob.glob(dir + "/*.png")
    logging.info("Detected %d images" , len(images))

    #limgs = [load_image(_) for _ in tqdm.tqdm(images,desc="Loading...")]
    logging.info("Done loading images   .")

    return images
def check_dataset_Image_size(image):
    return image.shape

def iskMultiInstance(prediction):
    return len(prediction) > 1

def pred2pose(prediction):
    """
    Extracts pose from prediction dictionary
    """
    
    return np.array(prediction['R']),np.array(prediction['t'])
def loadProjectionPoints(file):

    with open(file,'r') as f:
        data = json.load(f)
 
    return np.array([[data['3'][i][0]["x"],data['3'][i][0]["y"],data['3'][i][0]["z"]] for i in data["3"]])
def loadModelFaces(modelData):

    triangles3D = []
    for f in modelData['face']:
        vs = []
        for indice in f[0]:
            #print(indice)
            v = point3D(modelData["vertex"][indice][0], 
                            modelData["vertex"][indice][1], 
                            modelData["vertex"][indice][2])
            vs.append(v)
        tr = triangle3D(vs[0], 
                        vs[1], 
                        vs[2])       
        triangles3D.append(tr)

    return triangles3D
def load3DModel(modelpath):
    data = PlyData.read(modelpath)

    x = np.array(data['vertex']['x'])
    y = np.array(data['vertex']['y'])
    z = np.array(data['vertex']['z'])
    points = np.column_stack((x, y, z))

    return points, loadModelFaces(data)

def loadDecimatedModels(path):

    models= []
    for md in glob.glob(path + "/*.ply"):
        idx = int(os.path.basename(md).split("_")[1])
        models.append([idx,load3DModel(md)])

    return np.array(models,dtype=object)

def boundingBoxPerInstance(rot,tr,obj_points,K,obj_id):
    """Computes the bounding box of each instance present in the image
    by projecting some known 3D points of the model to the image and finding 
    the Lower-Left and Upper-Right points of the boudning box by finding the 
    maximum and minimum of the projected points in image coordinates.

    Args:
        rot (ND ARRAY): Rotation matrix 3 x 3
        tr (ND ARRAY): Translation vector 1 x 3
        obj_points (ND ARRAY): The loaded known 3D points of the model
        K (ND ARRAY): The calibration matrix
        obj_id (int): The object's id

    Returns:
        tuple: Lower-Left,Upper-Right points of the computed bounding box
    """
    t = time.time()
    result,_= cv.projectPoints(obj_points,
                              rot,
                              tr,
                              cameraMatrix=K,
                              distCoeffs=None)
    #print(f"Project points took : {time.time() - t}")
    # calculate the lower-left and upper-right of the bounding bot
    # lower-left -> [minx,miny]
    LL = (result[:,...,0].min() , result[:,...,1].max())
    
    # upper-right -> [maxx,maxy]
    UR = (result[:,...,0].max() , result[:,...,1].min())

    return LL,UR

def projectModelFace(image,face,K,rot,tr):
    """Projectsa a model face (v1,v2,v3) -> vertices in 3D coordinates
    to their 2D projection on the image

    Args:
        image (np.array): Image to project the face
        face (triangle3D): Face
        K (np.array): Calibration matrix
        rot (np.array): Rotation matrix
        tr (np.array): Translation matrix

    Returns:
        np.array: Image with projected face
    """
   
    result,_= cv.projectPoints(face,
                            rot,
                            tr,
                            cameraMatrix=K,
                            distCoeffs=None)
   
    cv.fillPoly(image,[result.astype(np.int32)],color=255)
    return image  
 
def filderCorrespFromBoundingBox(obj_corrs,LL,UR):

    # the x,y coordinates of the correspodense lie inside the 2D
    # bouding box if LL.x < x < UR.x and LL.y < y < LL.y
    condition = (obj_corrs['coord_2d'][:,0] < UR[0]) & \
                (obj_corrs['coord_2d'][:,0] > LL[0]) & \
                (obj_corrs['coord_2d'][:,1] > UR[1]) & \
                (obj_corrs['coord_2d'][:,1] < LL[1]) 
    
    return len(np.argwhere(condition))
    

import numpy as np
import cv2 as cv


def match_inter_board_points(corners,mrk_3d,ids):

    
    total_points_2d,total_points_3d = [],[]
    if len(ids)>=2:
        for idx,marker in enumerate(ids):
            if marker[0] in range(18):
                print(f"Looking for marker {marker[0]}")
                marker_2D = corners[idx]
                # get marker 3d corner points
            
                marker_3D = mrk_3d[marker[0]]

                #print(marker_3D)
                #print(marker_2D)
                total_points_2d.append(marker_2D)
                total_points_3d.append(marker_3D)

    return np.array(total_points_3d).reshape(-1,3),np.array(total_points_2d).reshape(-1,2)

            
# def vis_object_confs(image,conf,savePath):
#     """
#     Helper function to plot data with associated colormap.
#     """
#     #colormaps = [ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])]


#     # colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
#     # my_cmap = [ListedColormap(colors, name="my_cmap")]

#     viridis = mpl.colormaps['viridis'].resampled(256)
#     newcolors = viridis(np.linspace(0, 1, 256))
#     pink = np.array([248/256, 24/256, 148/256, 1])
#     newcmp = [ListedColormap(newcolors)]

    
#     fig, axs = plt.subplots(1, 1, figsize=(1 * 2 + 2, 3),
#                             constrained_layout=True, squeeze=False)
#     for [ax, cmap] in zip(axs.flat, newcmp):
#         psm = ax.pcolormesh(image, cmap=cmap, rasterized=True, vmin=0, vmax=255)
#         fig.colorbar(psm, ax=ax)
#     plt.gca().invert_yaxis()
#     bbox = dict(boxstyle ="round", fc ="1.0")
#     plt.annotate("Avg conf: "+str(np.round(conf,3)),(190,170),bbox=bbox)
#     plt.savefig(savePath)
#     plt.close(fig)