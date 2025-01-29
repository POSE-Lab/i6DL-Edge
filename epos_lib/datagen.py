# Copyright 2020 Tomas Hodan (hodantom@cmp.felk.cvut.cz).
# Copyright 2018 The TensorFlow Authors All Rights Reserved.

"""Data preparation functions."""

import os
import numpy as np
import igl
from bop_toolkit_lib import config as config_bop
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from epos_lib import fragment
from absl import logging


class ObjectModelStore(object):
  """Stores 3D object models, their fragmentation and other relevant data."""

  def __init__(self,
               dataset_name,
               model_type,
               num_frags,
               models=None,
               models_igl=None,
               frag_centers=None,
               frag_sizes=None,
               prepare_for_projection=False):
    """Initializes the 3D object model store.

    Args:
      dataset_name: Dataset name ('tless', 'ycbv', 'lmo', etc.).
      model_type: Type of object models (see bop_toolkit_lib/dataset_params.py).
      num_frags: Number of surface fragments per object.
      models: Dictionary of models (see load_ply from bop_toolkit_lib/inout.py).
      models_igl: Dictionary of models in the IGL format.
      frag_centers: Fragment centers.
      frag_sizes: Fragment sizes defined as the length of the longest side of
        the 3D bounding box of the fragment.
      prepare_for_projection: Whether to prepare the models for projection to
        the model surface.
    """
    self.dataset_name = dataset_name
    self.model_type = model_type
    self.models = models
    self.models_igl = models_igl
    self.num_frags = num_frags
    self.frag_centers = frag_centers
    self.frag_sizes = frag_sizes
    self.prepare_for_projection = prepare_for_projection
    self.aabb_trees_igl = {}

    # Dataset-specific model parameters.
    self.dp_model = dataset_params.get_model_params(
      config_bop.datasets_path, dataset_name, model_type=model_type)
  @property
  def num_objs(self):
    return len(self.models)

  def load_models(self):
    """Loads 3D object models."""
    logging.info('Loading object models...')

    self.models = {}
    self.models_igl = {}
    for obj_id in self.dp_model['obj_ids']:
      model_fpath = self.dp_model['model_tpath'].format(obj_id=obj_id)
      self.models[obj_id] = inout.load_ply(model_fpath)

      if self.prepare_for_projection:
        # Read the model to the igl format.
        V = igl.eigen.MatrixXd(self.models[obj_id]['pts'])
        F = igl.eigen.MatrixXi(self.models[obj_id]['faces'].astype(np.int32))
        self.models_igl[obj_id] = {'V': V, 'F': F}

    logging.info('Loaded {} object models.'.format(len(self.models)))

  def fragment_models(self):
    """Splits the surface of 3D object models into fragments."""
    logging.info('Fragmenting object models...')

    if self.models is None:
      self.load_models()

    self.frag_centers = {}
    self.frag_sizes = {}
    for obj_id in self.dp_model['obj_ids']:
      logging.info('Fragmenting object {}...'.format(obj_id))

      if self.num_frags == 1:
        # Use the origin (the center of the object) as the fragment center in
        # the case of one fragment.
        num_pts = self.models[obj_id]['pts'].shape[0]
        self.frag_centers[obj_id] = np.array([[0., 0., 0.]])
        pt_frag_ids = np.zeros(num_pts)
      else:
        # Find the fragment centers by the furthest point sampling algorithm.
        assert(len(self.models[obj_id]['pts']) >= self.num_frags)
        self.frag_centers[obj_id], pt_frag_ids =\
          fragment.fragmentation_fps(self.models[obj_id]['pts'], self.num_frags)

      # Calculate fragment sizes defined as the length of the longest side of
      # the 3D bounding box of the fragment.
      self.frag_sizes[obj_id] = []
      for frag_id in range(self.num_frags):
        # Points (i.e. model vertices) belonging to the current fragment.
        frag_pts =\
          self.models[obj_id]['pts'][pt_frag_ids == frag_id]

        # Calculate the 3D bounding box of the fragment and its longest side.
        bb_size = np.max(frag_pts, axis=0) - np.min(frag_pts, axis=0)
        min_frag_size = 5.0  # 5 mm.
        frag_size = max(np.max(bb_size), min_frag_size)
        self.frag_sizes[obj_id].append(frag_size)

      self.frag_sizes[obj_id] = np.array(self.frag_sizes[obj_id])

    logging.info('Object models fragmented.')

  def project_pts_to_model(self, pts, obj_id):
    """Projects 3D points to the model of the specified object.

    Args:
      pts: 3D points to project.
      obj_id: ID of the object model to which the points are projected.

    Returns:
      3D points projected to the model surface.
    """
    # Build AABB tree.
    if obj_id not in self.aabb_trees_igl:
      self.aabb_trees_igl[obj_id] = igl.AABB()
      self.aabb_trees_igl[obj_id].init(
        self.models_igl[obj_id]['V'], self.models_igl[obj_id]['F'])

    # Query points.
    P = igl.eigen.MatrixXd(pts)

    # For each query point, find the closest vertex on the model surface.
    sqrD = igl.eigen.MatrixXd()
    I = igl.eigen.MatrixXi()
    C = igl.eigen.MatrixXd()
    self.aabb_trees_igl[obj_id].squared_distance(
      self.models_igl[obj_id]['V'], self.models_igl[obj_id]['F'], P, sqrD, I, C)

    return np.array(C)