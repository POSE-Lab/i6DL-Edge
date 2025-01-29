import numpy as np
import os
import imageio
from PIL import Image as ImagePIL
from profiler import Profiler

def save_profiling(profiler: Profiler,savePath): #TODO store
    profiler.write_json_report(savePath)

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

def make_test_dirs(parent_path,timestamp,objID,profiler): #TODO store
# dynamically create test directorys based on the number of service calls
    profiler.addTrackItem("make_test_dirs")
    profiler.start("make_test_dirs")
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
    profiler.stop("make_test_dirs")
    return current_working_dir
def save_Image(path,img,timestamp,objID,profiler): #TODO store

    profiler.addTrackItem("save_image")
    profiler.start("save_image")

    print("Saving image at ", path+"/Test_"+str(timestamp)+"/Obj_"+str(objID)+"/image_raw.png")
    imageio.imwrite(path+"/Test_"+str(timestamp)+"/Obj_"+str(objID)+"/image_raw.png", img)

    profiler.stop("save_image")
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
    #print(inliers_indices.shape)
    #print("INLIERS",inliers)
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
    #   print("TIMES CALLED :",im_id)
    with open(corr_path, 'w') as f:
        f.write(txt)
