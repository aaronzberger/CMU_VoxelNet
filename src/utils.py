from __future__ import division
import os
import json
import math
import errno
import time

import numpy as np

from config import base_dir


def load_config(config_name=None):
    '''
    Load the configuration json file

    Returns:
        config (dict): Python dictionary of hyperparameter name-value pairs
    '''
    if config_name is None:
        config_name = 'config'
    path = os.path.join(base_dir, '{}.json'.format(config_name))

    with open(path) as file:
        config = json.load(file)

    return config


def mkdir_p(path):
    '''
    Create a directory at a given path if it does not already exist

    Parameters:
        path (string): the full os.path location for the directory
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_model_path(name):
    '''
    Get the path to save the model dict

    Parameters:
        name (any): name of the epoch (can be a number or token)
    '''
    config = load_config()
    save_dir = os.path.join(base_dir, 'save')

    mkdir_p(save_dir)

    if name is None:
        name = config['resume_from']

    return os.path.join(save_dir, str(name)+'epoch')


def get_num_voxels():
    '''
    Get the number of voxels in every dimension

    Returns:
        voxel_W (float): number of X voxels
        voxel_H (float): number of Y voxels
        voxel_D (float): number of Z voxels
    '''
    config = load_config()

    # Calculate the size of the voxel grid in every dimensions
    voxel_W = math.ceil(
        (config['pcl_range']['X2'] - config['pcl_range']['X1'])
        / config['voxel_size']['W'])
    voxel_H = math.ceil(
        (config['pcl_range']['Y2'] - config['pcl_range']['Y1'])
        / config['voxel_size']['H'])
    voxel_D = math.ceil(
        (config['pcl_range']['Z2'] - config['pcl_range']['Z1'])
        / config['voxel_size']['D'])

    return voxel_W, voxel_H, voxel_D


def get_anchors():
    '''
    Generate the anchors

    Returns:
        arr: list of anchors in the form [x,y,z,h,w,l,r]
    '''
    config = load_config()
    voxel_W, voxel_H, _ = get_num_voxels()

    # Make the anchor grid (center notation)
    x = np.linspace(config['pcl_range']['X1'] + config['voxel_size']['W'],
                    config['pcl_range']['X2'] - config['voxel_size']['W'],
                    voxel_W//2)
    y = np.linspace(config['pcl_range']['Y1'] + config['voxel_size']['H'],
                    config['pcl_range']['Y2'] - config['voxel_size']['H'],
                    voxel_H//2)

    # Get the xs and ys for the grid
    cx, cy = np.meshgrid(x, y)

    # Anchors only move in X and Y, not Z (BEV)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)

    # We only use one anchor size (See 3.1)
    cz = np.ones_like(cx) * -1.0
    width = np.ones_like(cx) * 1.6
    length = np.ones_like(cx) * 3.9
    height = np.ones_like(cx) * 1.56

    # We use two rotations: 0 and 90 deg (See 3.1)
    rotation = np.ones_like(cx)
    rotation[..., 0] = 0
    rotation[..., 1] = np.pi / 2

    return np.stack([cx, cy, cz, height, width, length, rotation], axis=-1)


def filter_pointcloud(lidar, boxes3d=None, config=None):
    '''
    Crop a lidar pointcloud to the dimensions specified in config json

    Parameters:
        lidar (arr): the point cloud
        boxes3d (arr): Ground truth boxes to crop

    Returns:
        arr: cropped point cloud
        arr: cropped ground truth boxes
    '''
    config = load_config(config)

    x_pts = lidar[:, 0]
    y_pts = lidar[:, 1]
    z_pts = lidar[:, 2]

    # Determine indexes of valid, in-bound points
    valid_x = np.where((x_pts >= config['pcl_range']['X1'])
                       & (x_pts < config['pcl_range']['X2']))[0]
    valid_y = np.where((y_pts >= config['pcl_range']['Y1'])
                       & (y_pts < config['pcl_range']['Y2']))[0]
    valid_z = np.where((z_pts >= config['pcl_range']['Z1'])
                       & (z_pts < config['pcl_range']['Z2']))[0]

    # Combine the index arrays
    valid_xy = np.intersect1d(valid_x, valid_y)
    valid_xyz = np.intersect1d(valid_xy, valid_z)

    # Also crop the 3d boxes if provided
    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= config['pcl_range']['X1']) \
            & (boxes3d[:, :, 0] < config['pcl_range']['X2'])
        box_y = (boxes3d[:, :, 1] >= config['pcl_range']['Y1']) \
            & (boxes3d[:, :, 1] < config['pcl_range']['Y2'])
        box_z = (boxes3d[:, :, 2] >= config['pcl_range']['Z1']) \
            & (boxes3d[:, :, 2] < config['pcl_range']['Z2'])
        box_xyz = np.sum(box_x & box_y & box_z, axis=1)

        return lidar[valid_xyz], boxes3d[box_xyz > 0]

    return lidar[valid_xyz]


def load_kitti_calib(calib_file):
    """
    Retrieve the transforms from the calibration file

    Parameters:
        calib_file (str): calibration file path

    Returns:
        dict: dict containing necessary transforms
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def box3d_cam_to_velo(box3d, Tr=None):
    '''
    Transform bounding boxes from center to corner notation
    and transform to velodyne frame

    Parameters:
        box3d (arr): the bouning box in center notation
        Tr (arr): the transform from camera to velodyne

    Returns:
        arr: bounding box in corner notation
    '''

    def project_cam2velo(cam, Tr):
        if Tr is None:
            return cam[:3].reshape(1, 3)
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle

        return angle

    # KITTI labels are formatted [hwlxyzr]
    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]

    # Position in labels are in cam coordinates. Transform to lidar coords
    cam = np.expand_dims(np.array([tx, ty, tz, 1]), 1)
    translation = project_cam2velo(cam, Tr)

    # Very similar code as in box3d_center_to_corner in conversions.py
    # Create the bounding box outline (to be transposed)
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [0, 0, 0, 0, h, h, h, h]])

    rotation = ry_to_rz(ry)

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    corner_box = corner_box.transpose()

    return corner_box.astype(np.float32)


def load_kitti_label(label_file, Tr):
    '''
    Load the labels for a specific image

    Parameters:
        label_file (str): label file full path
        Tr (arr): velodyne to camera transform

    Returns:
        arr: array containing GT boxes in the correct format
    '''
    config = load_config()

    with open(label_file, 'r') as f:
        lines = f.readlines()

    gt_boxes3d_corner = []

    for j in range(len(lines)):
        obj = lines[j].strip().split(' ')

        # Ensure the GT class is one we're using
        if obj[0].strip() not in config['class_list']:
            continue

        # Transform label into coordinates of 8 points that make up the bbox
        box3d_corner = box3d_cam_to_velo(obj[8:], Tr)

        gt_boxes3d_corner.append(box3d_corner)

    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1, 8, 3)

    return gt_boxes3d_corner


class Timer:
    '''
    Simple timer class to keep track of mutliple running timers.
    Just used for debugging.
    '''
    def __init__(self, num=10):
        self.timers = [time.time()] * num

    def start(self, num=0):
        self.check_list(num)
        self.timers[num] = time.time()

    def stop(self, num=0):
        if num > len(self.timers) - 1:
            raise ValueError('Timer {} doesn\'t exist'.format(num))
        end = time.time() - self.timers[num]
        self.timers[num] = time.time()
        return end

    def get(self, num=0):
        self.check_list(num)
        return time.time() - self.timers[num]

    def check_list(self, num):
        '''Double the length of the timer array if we need it'''
        if num > len(self.timers) - 1:
            self.timers = self.timers + [time.time()] * len(self.timers)
