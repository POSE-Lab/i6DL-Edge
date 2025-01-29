import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import numpy as np
from PIL import Image
import time
import sys, os
import glob
from epos_lib import common  # utility functions
from epos_lib import misc  # for the image resize
import sys
from epos_lib import corresp
#from bop_toolkit_lib import inout
#from epos_lib import vis
import time
import cv2
from epos_lib import datagen  # utility functions
import pyprogressivex
#from bop_toolkit_lib import visualization
import pickle
#from bop_toolkit_lib import dataset_params  # dataset specs
from epos_lib import config  # config flags and constants
import glob
import json
from absl import app
from absl import flags
from absl import logging
from confidence import plot_examples, confidense_calc, plot_pose_mask

np.set_printoptions(precision=10, threshold=sys.maxsize,suppress=True)
# parser = argparse.ArgumentParser()
# parser.add_argument("--engine",required=True,type=str)
# parser.add_argument("--img_path",required=True,type=str)
# parser.add_argument("--multiImage",action="store_true")
# parser.add_argument("--verboose",action="store_true")
# parser.add_argument("--benchmark",action="store_true")
# parser.add_argument("--warmup",required=True,type=int)
# args = parser.parse_args()

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'method','',
    'Optimization method'
)
flags.DEFINE_integer(
    'request_object',None,
    'Request object id'
)
flags.DEFINE_string(
    'engine', '',
    "Engine"
)
flags.DEFINE_string(
    'img_path', '',
    "Image path"
)
flags.DEFINE_boolean(
    'multiImage', True,
    "Image path"
)
flags.DEFINE_boolean(
    'verbose', False,
    "Image path"
)
flags.DEFINE_boolean(
    'run_for_mult_images', False,
    'Run detect for mulitple images after warmup session'
)
flags.DEFINE_string(
    'master', '',
    'BNS name of the tensorflow server')
flags.DEFINE_boolean(
    'cpu_only', False,
    'Whether to run the inference on CPU only.')
flags.DEFINE_string(
    'task_type', common.LOCALIZATION,  # LOCALIZATION, DETECTION
    'Type of the 6D object pose estimation task.')
flags.DEFINE_list(
    'infer_tfrecord_names', None,
    'Names of tfrecord files (without suffix) used for inference.')
flags.DEFINE_integer(
    'infer_max_height_before_crop', '1280',
    'Maximum image height before cropping (the image is downscaled if larger).')
flags.DEFINE_list(
    'infer_crop_size', '720,1280',
    'Image size [height, width] during inference.')
flags.DEFINE_string(
    'checkpoint_name', None,
    'Name of the checkpoint to evaluate (e.g. "model.ckpt-1000000"). The latest '
    'available checkpoint is used if None.')
flags.DEFINE_boolean(
    'project_to_surface', False,
    'Whether to project the predicted 3D locations to the object model.')
flags.DEFINE_boolean(
    'save_estimates', True,
    'Whether to save pose estimates in format expected by the BOP Challenge.')
flags.DEFINE_boolean(
    'save_corresp', True,
    'Whether to save established correspondences to text files.')
flags.DEFINE_string(
    'infer_name', None,
    'Name of the inference used in the filename of the saved estimates.')

# Pose fitting parameters.
flags.DEFINE_string(
    'fitting_method', common.PROGRESSIVE_X,  # PROGRESSIVE_X, OPENCV_RANSAC
    'Pose fitting method.')
flags.DEFINE_float(
    'inlier_thresh', 4.0,
    'Tau_r in the CVPR 2020 paper. Inlier threshold [px] on the '
    'reprojection error.')
flags.DEFINE_float(
    'neighbour_max_dist', 20.0,
    'Tau_d in the CVPR 2020 paper.')
flags.DEFINE_float(
    'min_hypothesis_quality', 0.5,
    'Tau_q in the CVPR 2020 paper')
flags.DEFINE_float(
    'required_progx_confidence', 0.5,
    'The required confidence used to calculate the number of Prog-X iterations.')
flags.DEFINE_float(
    'required_ransac_confidence', 1.0,
    'The required confidence used to calculate the number of RANSAC iterations.')
flags.DEFINE_float(
    'min_triangle_area', 0.0,
    'Tau_t in the CVPR 2020 paper.')
flags.DEFINE_boolean(
    'use_prosac', False,
    'Whether to use the PROSAC sampler.')
flags.DEFINE_integer(
    'max_model_number_for_pearl', 5,
    'Maximum number of instances to optimize by PEARL. PEARL is turned off if '
    'there are more instances to find.')
flags.DEFINE_float(
    'spatial_coherence_weight', 0.1,
    'Weight of the spatial coherence in Graph-Cut RANSAC.')
flags.DEFINE_float(
    'scaling_from_millimeters', 0.1,
    'Scaling factor of 3D coordinates when'
    '0.1 will convert mm to cm. See the CVPR 2020 paper for details.')
flags.DEFINE_float(
    'max_tanimoto_similarity', 0.9,
    'See the Progressive-X paper.')
flags.DEFINE_integer(
    'max_correspondences', None,
    'Maximum number of correspondences to use for fitting. Not applied if None.')
flags.DEFINE_integer(
    'max_instances_to_fit', None,
    'Maximum number of instances to fit. Not applied if None.')
flags.DEFINE_integer(
    'max_fitting_iterations', 400,
    'The maximum number of fitting iterations.')

# Visualization parameters.
flags.DEFINE_boolean(
    'vis', False,
    'Global switch for visualizations.')
flags.DEFINE_boolean(
    'vis_gt_poses', False,
    'Whether to visualize the GT poses.')
flags.DEFINE_boolean(
    'vis_pred_poses', True,
    'Whether to visualize the predicted poses.')
flags.DEFINE_boolean(
    'vis_gt_obj_labels', False,
    'Whether to visualize the GT object labels.')
flags.DEFINE_boolean(
    'vis_pred_obj_labels', True,
    'Whether to visualize the predicted object labels.')
flags.DEFINE_boolean(
    'vis_pred_obj_confs', False,
    'Whether to visualize the predicted object confidences.')
flags.DEFINE_boolean(
    'vis_gt_frag_fields', False,
    'Whether to visualize the GT fragment fields.')
flags.DEFINE_boolean(
    'vis_pred_frag_fields', False,
    'Whether to visualize the predicted fragment fields.')
# ------------------------------------------------------------------------------
def save_EPOS_pose(path,pos,orie,objid, K_mat):
    orie = np.array(orie).reshape((3,3))
    _t = np.vstack((orie,np.array([0.0,0.0,0.0]).reshape((1,3))))
    print(_t)
    _f = np.array(pos).reshape((3,1))
    _f = np.append(_f,[1.0])
    print(_f)
    _f = np.hstack((_t,_f.reshape((4,1))))

    filename = path+"/Obj_"+str(objid)+"/pose.txt"
    print("saving pose at ", filename)
    np.savetxt(filename, _f, newline=',\n',delimiter=',',fmt='%f')
    file = open(filename, 'a')
    np.savetxt(file, K_mat, newline=', ',delimiter=',',fmt='%f')
    file.close()
    #f.write(str(orie))

def avg(list):
    " average of list of dicrtionarys"
    sum1=0
    sum2=0
    sum3=0
    for item in list:
        sum1+=item['prediction']
        sum2+=item['establish_corr']
        sum3+=item['fitting']
    return sum1/len(list), sum2/len(list), sum3/len(list)
def visualize(
        image, predictions, pred_poses, im_ind, crop_size, output_scale,
        model_store, renderer, vis_dir):
    """Visualizes estimates from one image.

    Args:
      samples: Dictionary with input data.
      predictions: Dictionary with predictions.
      pred_poses: Predicted poses.
      im_ind: Image index.    fx = 640.9191284179688  # * im_scale
    fy = 639.4614868164062  # * im_scale
    cx = 631.1490478515625  # * im_scale
    cy = 363.1187744140625  # * im_scale
      vis_dir: Directory where the visualizations will be saved.
    """


    # Size of a visualization grid tile.
    tile_size = (300, 225)

    # Extension of the saved visualizations ('jpg', 'png', etc.).
    vis_ext = 'jpg'

    # Font settings.
    font_size = 10
    font_color = (0.8, 0.8, 0.8)

    # Intrinsics.
     # Intrinsic parameters.
    fx = 634.364  # * im_scale
    fy = 637.801  # * im_scale
    cx = 633.635  # * im_scale
    cy = 364.958  # * im_scale

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    output_K = K * output_scale
    output_K[2, 2] = 1.0

    # Tiles for the grid visualization.
    tiles = []

    # Size of the output fields.
    output_size =\
        int(output_scale * crop_size[0]), int(output_scale * crop_size[1])

    # Prefix of the visualization names.
    vis_prefix = '{:06d}'.format(im_ind)

    # Input RGB image.
    rgb = np.array(Image.open(image))
    vis_rgb = visualization.write_text_on_image(
        misc.resize_image_py(rgb, tile_size).astype(np.uint8),
        [{'name': '', 'val': 'input', 'fmt': ':s'}],
        size=font_size, color=font_color)
    tiles.append(vis_rgb)

    # Visualize the estimated poses.
    if FLAGS.vis_pred_poses:
        vis_pred_poses = vis.visualize_object_poses(
            rgb, K, pred_poses, renderer)
        vis_pred_poses = visualization.write_text_on_image(
            misc.resize_image_py(vis_pred_poses, tile_size),
            [{'name': '', 'val': 'pred poses', 'fmt': ':s'}],
            size=font_size, color=font_color)
        tiles.append(vis_pred_poses)

    # Predicted object labels.
    if FLAGS.vis_pred_obj_labels:
        obj_labels = np.squeeze(predictions[common.PRED_OBJ_LABEL][0])
        obj_labels = obj_labels[:crop_size[1], :crop_size[0]]
        obj_labels = vis.colorize_label_map(obj_labels)
        obj_labels = visualization.write_text_on_image(
            misc.resize_image_py(obj_labels.astype(np.uint8), tile_size),
            [{'name': '', 'val': 'predicted obj labels', 'fmt': ':s'}],
            size=font_size, color=font_color)
        tiles.append(obj_labels)

    # Predicted object confidences.
    if FLAGS.vis_pred_obj_confs:
        num_obj_labels = predictions[common.PRED_OBJ_CONF].shape[-1]
        for obj_label in range(num_obj_labels):
            obj_confs = misc.resize_image_py(np.array(
                predictions[common.PRED_OBJ_CONF][0, :, :, obj_label]), tile_size)
            obj_confs = (255.0 * obj_confs).astype(np.uint8)
            obj_confs = np.dstack([obj_confs, obj_confs, obj_confs])  # To RGB.
            obj_confs = visualization.write_text_on_image(
                obj_confs, [{'name': 'cls', 'val': obj_label, 'fmt': ':d'}],
                size=font_size, color=font_color)
            tiles.append(obj_confs)

    # Visualization of predicted fragment fields.
    if FLAGS.vis_pred_frag_fields:
        vis.visualize_pred_frag(
            frag_confs=predictions[common.PRED_FRAG_CONF][0],
            frag_coords=predictions[common.PRED_FRAG_LOC][0],
            output_size=output_size,
            model_store=model_store,
            vis_prefix=vis_prefix,
            vis_dir=vis_dir,
            vis_ext=vis_ext)

    # Build and save a visualization grid.
    grid = vis.build_grid(tiles, tile_size)
    grid_vis_path = os.path.join(
        vis_dir, '{}_grid.{}'.format(vis_prefix, vis_ext))
    inout.save_im(grid_vis_path, grid)

# def save_correspondences(
#         scene_id, im_id, im_ind, obj_id, image_path, K, obj_pred, pred_time,
#         infer_name, obj_gt_poses, infer_dir):

#     # Add meta information.
#     txt = '# Corr format: u v x y z px_id frag_id conf conf_obj conf_frag\n'
#     txt += '{}\n'.format(image_path)
#     txt += '{} {} {} {}\n'.format(scene_id, im_id, obj_id, pred_time)

#     # Add intrinsics.
#     for i in range(3):
#         txt += '{} {} {}\n'.format(K[i, 0], K[i, 1], K[i, 2])

#     # Add ground-truth poses.
#     txt += '{}\n'.format(len(obj_gt_poses))
#     for pose in obj_gt_poses:
#         for i in range(3):
#             txt += '{} {} {} {}\n'.format(
#                 pose['R'][i, 0], pose['R'][i, 1], pose['R'][i, 2], pose['t'][i, 0])

#     # Sort the predicted correspondences by confidence.
#     sort_inds = np.argsort(obj_pred['conf'])[::-1]
#     px_id = obj_pred['px_id'][sort_inds]
#     frag_id = obj_pred['frag_id'][sort_inds]
#     coord_2d = obj_pred['coord_2d'][sort_inds]
#     coord_3d = obj_pred['coord_3d'][sort_inds]
#     conf = obj_pred['conf'][sort_inds]
#     conf_obj = obj_pred['conf_obj'][sort_inds]
#     conf_frag = obj_pred['conf_frag'][sort_inds]

#     # Add the predicted correspondences.
#     pred_corr_num = len(coord_2d)
#     txt += '{}\n'.format(pred_corr_num)
#     for i in range(pred_corr_num):
#         txt += '{} {} {} {} {} {} {} {} {} {}\n'.format(
#             coord_2d[i, 0], coord_2d[i, 1],
#             coord_3d[i, 0], coord_3d[i, 1], coord_3d[i, 2],
#             px_id[i], frag_id[i], conf[i], conf_obj[i], conf_frag[i])

#     # Save the correspondences into a file.
#     corr_suffix = infer_name
#     if corr_suffix is None:
#         corr_suffix = ''
#     else:
#         corr_suffix = '_' + corr_suffix

#     corr_path = os.path.join(
#         infer_dir, 'corr{}'.format(corr_suffix),
#         '{:06d}_corr_{:02d}.txt'.format(im_ind, obj_id))
#     with open(corr_path, 'w') as f:
#         f.write(txt)
def save_correspondences(
        scene_id, im_id, obj_id, image_path, K, obj_pred, pred_time,
        infer_name, obj_gt_poses, infer_dir,inliers_indices):

    # Add meta information.
    txt = '# Corr format: u v x y z px_id frag_id conf conf_obj conf_frag\n'
    txt += '{}\n'.format(image_path)
    print(inliers_indices.shape)
    inliers = np.count_nonzero(inliers_indices == 1)
    total = len(obj_pred['coord_2d'][np.argsort(obj_pred['conf'])[::-1]])
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
    sort_inds = np.argsort(obj_pred['conf'])[::-1]
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
        infer_dir, 'corr{}'.format(corr_suffix),str(im_id)+"_corr")
    print("TIMES CALLED :",im_id)
    with open(corr_path, 'w') as f:
        f.write(txt)
def process_image(K,predTime,image,predictions, im_id, scene_id,crop_size, output_scale, model_store,
        renderer, task_type, infer_name, infer_dir, vis_dir):
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
        min_obj_conf=FLAGS.corr_min_obj_conf,
        min_frag_rel_conf=FLAGS.corr_min_frag_rel_conf,
        project_to_surface=FLAGS.project_to_surface,
        only_annotated_objs=(task_type == common.LOCALIZATION))
    run_times['establish_corr'] = time.time() - time_start

    # PnP-RANSAC to estimate 6D object poses from the correspondences.
    time_start = time.time()
    poses = []
    for obj_id, obj_corr in corr.items():
        # tf.compat.v1.logging.info(
        #   'Image path: {}, obj: {}'.format(samples[common.IMAGE_PATH][0], obj_id))

        # Number of established correspondences.
        num_corrs = obj_corr['coord_2d'].shape[0]

        # Skip the fitting if there are too few correspondences.
        min_required_corrs = 6
        if num_corrs < min_required_corrs:
            continue

        # The correspondences need to be sorted for PROSAC.
        if FLAGS.use_prosac:    
           
            sorted_inds = np.argsort(obj_corr['conf'])[::-1]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][sorted_inds]

        # Select correspondences with the highest confidence.
        if FLAGS.max_correspondences is not None \
                and num_corrs > FLAGS.max_correspondences:
            # Sort the correspondences only if they have not been sorted for PROSAC.
            if FLAGS.use_prosac:
                keep_inds = np.arange(num_corrs)
            else:
                keep_inds = np.argsort(obj_corr['conf'])[::-1]
            keep_inds = keep_inds[:FLAGS.max_correspondences]
            for key in obj_corr.keys():
                obj_corr[key] = obj_corr[key][keep_inds]

        # Save the established correspondences (for analysis).
        if FLAGS.save_corresp:
            obj_gt_poses = []
            if gt_poses is not None:
                obj_gt_poses = [x for x in gt_poses if x['obj_id'] == obj_id]
            pred_time = float(np.sum(list(run_times.values())))
            image_path = "test"
            save_correspondences(
                scene_id, im_id, obj_id, image_path, K, obj_corr, pred_time,
                "", obj_gt_poses,"/home/panos/Documents/cors_res",inlier_indices)

        # Make sure the coordinates are saved continuously in memory.
        coord_2d = np.ascontiguousarray(
            obj_corr['coord_2d'].astype(np.float64))
        coord_3d = np.ascontiguousarray(
            obj_corr['coord_3d'].astype(np.float64))
        print(len(coord_2d))
        if FLAGS.fitting_method == common.PROGRESSIVE_X:
            # If num_instances == 1, then only GC-RANSAC is applied. If > 1, then
            # Progressive-X is applied and up to num_instances poses are returned.
            # If num_instances == -1, then Progressive-X is applied and all found
            # poses are returned.
            if task_type == common.LOCALIZATION:
                num_instances = 1
            else:
                num_instances = -1

            if FLAGS.max_instances_to_fit is not None:
                num_instances = min(num_instances, FLAGS.max_instances_to_fit)

            pose_ests, inlier_indices, pose_qualities = pyprogressivex.find6DPoses(
                x1y1=coord_2d,
                x2y2z2=coord_3d,
                K=K,
                threshold=FLAGS.inlier_thresh,
                neighborhood_ball_radius=FLAGS.neighbour_max_dist,
                spatial_coherence_weight=FLAGS.spatial_coherence_weight,
                scaling_from_millimeters=FLAGS.scaling_from_millimeters,
                max_tanimoto_similarity=FLAGS.max_tanimoto_similarity,
                max_iters=FLAGS.max_fitting_iterations,
                conf=FLAGS.required_progx_confidence,
                proposal_engine_conf=FLAGS.required_ransac_confidence,
                min_coverage=FLAGS.min_hypothesis_quality,
                min_triangle_area=FLAGS.min_triangle_area,
                min_point_number=6,
                max_model_number=num_instances,
                max_model_number_for_optimization=FLAGS.max_model_number_for_pearl,
                use_prosac=FLAGS.use_prosac,
                log=False)
            
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

        elif FLAGS.fitting_method == common.OPENCV_RANSAC:
            print("IAHERE")
            # This integration of OpenCV-RANSAC can estimate pose of only one object
            # instance. Note that in Table 3 of the EPOS CVPR'20 paper, the scores
            # for OpenCV-RANSAC were obtained with integrating cv2.solvePnPRansac
            # in the Progressive-X scheme (as the other methods in that table).
            pose_est_success, r_est, t_est, inliers = cv2.solvePnPRansac(
                objectPoints=coord_3d,
                imagePoints=coord_2d,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=FLAGS.max_fitting_iterations,
                reprojectionError=FLAGS.inlier_thresh,
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
                'Unknown pose fitting method ({}).'.format(FLAGS.fitting_method))

    run_times['fitting'] = time.time() - time_start
    run_times['total'] = np.sum(list(run_times.values()))

    # Add the total time to each pose.
    for pose in poses:
        pose['time'] = run_times['total']

    # Visualization.
    if FLAGS.vis:
        visualize(
            image=image,
            predictions=predictions,
            pred_poses=poses,
            im_ind=im_id,
            crop_size=crop_size,
            output_scale=output_scale,
            model_store=model_store,
            renderer=renderer,
            vis_dir=vis_dir)

    return poses, run_times

def load_img(imagepath):
    img = np.array(Image.open(imagepath)).astype(np.float32)
    return img
def load_engine(engine_path,runtime):
    # Deserialize engine from file
    with open(FLAGS.engine,"rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    return engine
def load_pil_image(imagepath):
    return Image.open(imagepath)
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

        eval_start_time = time.time()

        cuda.memcpy_htod_async(self.input_gpu,load_img(image),self.stream)
        #Copy inouts
        self.context.execute_async_v2(bindings=[int(self.input_gpu)] + [int(outp) for outp in self.gpu_outputs] , 
            stream_handle=self.stream.handle)

        for i in range(4):
            cuda.memcpy_dtoh_async(self.cpu_outputs[i], self.gpu_outputs[i], self.stream)

        self.stream.synchronize()

        eval_time_elapsed = time.time() - eval_start_time

        self.predictions={'pred_frag_conf':self.cpu_outputs[2],
                    'pred_frag_loc':self.cpu_outputs[1],
                    'pred_obj_conf':self.cpu_outputs[0],
                    'pred_obj_label':np.argmax(self.cpu_outputs[0],axis=-1)}

        if FLAGS.verbose:

            with open("test_label.txt",'w') as f:
                f.write(str(self.predictions['pred_obj_label']))
            #print(predictions)
            print("Inference took : ",eval_time_elapsed)
        
        print(eval_time_elapsed)

        return self.predictions,eval_time_elapsed
    
    def visualize(self,model_id,original_image):
        
        np_image = np.array(self.cpu_outputs[0])
        image_mask = np.array(self.cpu_outputs[3])

        obj_labels = np.squeeze(self.predictions['pred_obj_label'])
        print(obj_labels)
        print('SHAPEE')
        image_mask = np.reshape(obj_labels,(180,320)) * 255
        print(image_mask)
        #print(image_mask.shape)
        plot_pose_mask(image_mask.astype(np.uint8),original_image)
        #np_image= np_image*255
        #np_image[np_image!=0] = 255
        #print(np_image)
        print(np_image.shape[-1])
        original = np.reshape(np_image,(180,320,3))
        im = np.reshape(np_image,(180,320,3))*255
        image = Image.fromarray(im.astype(np.uint8))
        
        red = Image.fromarray(im.astype(np.uint8))
        #red.show()

        #image.save("/home/panos/Documents/carObj1_confidence/softmax.png")

        arg = im.astype(np.uint8)
        #based on object label
        arg = np.dstack([arg[:,:,model_id], arg[:,:,model_id], arg[:,:,model_id]])
        arg_image = Image.fromarray(arg)
        la,obj_pixels = confidense_calc(original,model_id)
        print(la)
        with open("test.txt",'w') as f:
             f.write(str(la))
        if la!=-999:
            plot_examples(arg_image.convert("L"),conf=la,savePath="/home/panos/code/epos/store/tf_models/crfAndLab12_1280_720_reannot/vis/00000"+str(model_id)+".png")
        #arg_image.convert("L").show()

        #arg_image.save("/home/panos/Documents/carObj1_confidence/argmax.png")

def main(argv):

     # Model folder.
    model_dir = os.path.join(config.TF_MODELS_PATH, FLAGS.model)

    # Update flags with parameters loaded from the model folder.
    common.update_flags(os.path.join(model_dir, common.PARAMS_FILENAME))

    # Print the flag values.
    common.print_flags()

    # Folder from which the latest model checkpoint will be loaded.
    checkpoint_dir = os.path.join(model_dir, 'train')

    # Folder for the inference output.
    infer_dir = os.path.join(model_dir, 'infer')
    try:
        os.makedirs(infer_dir)
    except FileExistsError:
        # directory already exists
        pass

    # Folder for the visualization output.
    vis_dir = os.path.join(model_dir, 'vis')
    try:
        os.makedirs(vis_dir)
    except FileExistsError:
        # directory already exists
        pass

    if FLAGS.upsample_logits:
        # The stride is 1 if the logits are upsampled to the input resolution.
        output_stride = 1
    else:
        assert (len(FLAGS.decoder_output_stride) == 1)
        output_stride = FLAGS.decoder_output_stride[0]

    renderer = None
    if FLAGS.vis:
        
        renderer = bop_renderer.Renderer()
        renderer.init(1280, 720)

        # load model info

        model_type_vis = None
        dp_model = dataset_params.get_model_params(
            config.BOP_PATH, 'carObj1', model_type=model_type_vis)

        for obj_id in dp_model['obj_ids']:
            path = dp_model['model_tpath'].format(obj_id=obj_id)
            renderer.add_object(obj_id, path)

        

    frag_path = os.path.join(model_dir, 'fragments.pkl')
    if os.path.exists(frag_path):
        

        with open(frag_path, 'rb') as f:
            fragments = pickle.load(f)
            frag_centers = fragments['frag_centers']
            frag_sizes = fragments['frag_sizes']

        # Check if the loaded fragmentation is valid.
        for obj_id in frag_centers.keys():
            if frag_centers[obj_id].shape[0] != FLAGS.num_frags\
                    or frag_sizes[obj_id].shape[0] != FLAGS.num_frags:
                raise ValueError('The loaded fragmentation is not valid.')

    else:
        logging.info(
            'Fragmentation does not exist (expected file: {}).'.format(frag_path))
        logging.info('Calculating fragmentation...')

        model_type_frag_str = 'cad'
        if model_type_frag_str is None:
            model_type_frag_str = 'original'
        logging.info('Type of models: {}'.format(model_type_frag_str))

    # Load 3D object models for fragmentation.
        model_store_frag = datagen.ObjectModelStore(
            dataset_name='carObj1',
            model_type='cad',
            num_frags=FLAGS.num_frags,
            prepare_for_projection=FLAGS.project_to_surface)

        # Fragment the 3D object models.
        model_store_frag.fragment_models()
        frag_centers = model_store_frag.frag_centers
        frag_sizes = model_store_frag.frag_sizes

    # Load 3D object models for rendering.
    model_store = datagen.ObjectModelStore(
        dataset_name='carObj1',
        model_type='cad',
        num_frags=FLAGS.num_frags,
        frag_centers=frag_centers,
        frag_sizes=frag_sizes,
        prepare_for_projection=FLAGS.project_to_surface)
    model_store.load_models()

    outputs_to_num_channels = common.get_outputs_to_num_channels(
        2, FLAGS.num_frags)

    trtEngine = TRT_engine(FLAGS.engine,"rgb_image_input:0")
    print("hallooooo")
    if FLAGS.multiImage:
        poses_all = []
        times=[]
        print("MutliImage enabled")
        #get scenes
        scenes = [s for s in os.listdir(FLAGS.img_path)]
        print(scenes)
        for scene in scenes:
            # read K from scene camera
            with open(os.path.join(FLAGS.img_path,scene)+"/Kmat.json",'r') as f:
                data = json.load(f)
                K = data['K']
            print(np.array(K).reshape((3,3)))
            images = [img for img in glob.glob(os.path.join(FLAGS.img_path,scene)+"/rgb"+"/*.png")]
            for img in sorted(images):
                print("Processing image "+img)
                predictions,predTime = trtEngine.predict(img)
                trtEngine.visualize(int(scene),img)
                poses,runtimes = process_image(
                        K=K,
                        predTime = predTime,
                        image=img,
                        predictions= predictions,
                        im_id=int(os.path.basename(img).split('.')[0]),
                        scene_id=int(scene),
                        crop_size=list(map(int, FLAGS.infer_crop_size)),
                        output_scale=(1.0 / output_stride),
                        model_store=model_store,
                        renderer=renderer,
                        task_type=FLAGS.task_type,
                        infer_name=FLAGS.infer_name,
                        infer_dir=infer_dir,
                        vis_dir=vis_dir)
                poses_all.append(poses[0])
                pos = poses[0]['t']
                orie = poses[0]['R'].flatten('C')
                save_EPOS_pose("/home/panos/Documents/est_poses",pos,orie,int(scene), K)
                times.append(runtimes)
                #print(poses)
        print("Inference done")
        avg_pred,avg_corr,avg_fitting = avg(times)
        print("Average prediction time per image: "+ str(avg_pred))
        print("Average establish correspondences time per image: "+ str(avg_corr))
        print("Average fitting time per image: "+ str(avg_fitting))
        if FLAGS.save_estimates:
            suffix = ''
            if FLAGS.infer_name is not None:
                suffix = '_{}'.format(FLAGS.infer_name)
            poses_path = os.path.join(
                infer_dir, 'method{}_carObj1-test_640_480{}.csv'.format(FLAGS.method,suffix))
            logging.info('Saving estimated poses to: {}'.format(poses_path))
            inout.save_bop_results(poses_path, poses_all, version='bop19')
    else:
        print("Single image")
        predictions,predTime = trtEngine.predict(FLAGS.img_path)
        #print(predictions)
        #trtEngine.visualize(2)
        
        K=[634.364,0.0,637.801,0.0,633.635,364.958,0.0,0.0,1.0]
        # with open("/home/panos/SPREADER_DATASET/carObj1/test_primesense/000002/Kmat.json",'r') as f:
        #         data = json.load(f)
        #         K = data['K']
        trtEngine.visualize(1,load_pil_image(FLAGS.img_path))
        poses,runtimes = process_image(
                        K=K,
                        predTime = predTime,
                        image=FLAGS.img_path,
                        predictions= predictions,
                        im_id=0,
                        scene_id=2,
                        crop_size=list(map(int, FLAGS.infer_crop_size)),
                        output_scale=(1.0 / output_stride),
                        model_store=model_store,
                        renderer=renderer,
                        task_type=FLAGS.task_type,
                        infer_name=FLAGS.infer_name,
                        infer_dir=infer_dir,
                        vis_dir=vis_dir)
        print(poses)
        print(runtimes)

if __name__ == "__main__":
    app.run(main)
                    
        # trtEngine.visualize()

    #TODO: visualizationtime