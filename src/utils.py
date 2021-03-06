from __future__ import division
import os
import json
import math
import errno
import time

import numpy as np

from config import base_dir, config


def load_config():
    '''
    Load the configuration json file
    '''
    with open(config) as file:
        config_dict = json.load(file)

    return config_dict


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
                    voxel_W // 2)
    y = np.linspace(config['pcl_range']['Y1'] + config['voxel_size']['H'],
                    config['pcl_range']['Y2'] - config['voxel_size']['H'],
                    voxel_H // 2)

    # x = np.linspace(config['pcl_range']['X1'], config['pcl_range']['X2'],
    #                 voxel_W)
    # y = np.linspace(config['pcl_range']['Y1'], config['pcl_range']['Y2'],
    #                 voxel_H)

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


def filter_pointcloud(lidar, boxes3d=None):
    '''
    Crop a lidar pointcloud to the dimensions specified in config json

    Parameters:
        lidar (arr): the point cloud
        boxes3d (arr): Ground truth boxes to crop

    Returns:
        arr: cropped point cloud
        arr: valid ground truth boxes
    '''
    config = load_config()

    x_pts = lidar[:, 0]
    y_pts = lidar[:, 1]
    z_pts = lidar[:, 2]

    # Determine indexes of valid, in-bound points
    lidar_x = np.where((x_pts >= config['pcl_range']['X1'])
                       & (x_pts < config['pcl_range']['X2']))[0]
    lidar_y = np.where((y_pts >= config['pcl_range']['Y1'])
                       & (y_pts < config['pcl_range']['Y2']))[0]
    lidar_z = np.where((z_pts >= config['pcl_range']['Z1'])
                       & (z_pts < config['pcl_range']['Z2']))[0]

    # Combine the index arrays
    lidar_valid_xyz = np.intersect1d(lidar_z, np.intersect1d(lidar_x, lidar_y))

    # Also crop the 3d boxes if provided
    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= config['pcl_range']['X1']) \
            & (boxes3d[:, :, 0] < config['pcl_range']['X2'])
        box_y = (boxes3d[:, :, 1] >= config['pcl_range']['Y1']) \
            & (boxes3d[:, :, 1] < config['pcl_range']['Y2'])
        box_z = (boxes3d[:, :, 2] >= config['pcl_range']['Z1']) \
            & (boxes3d[:, :, 2] < config['pcl_range']['Z2'])

        box_valid_xyz = np.sum(box_x & box_y & box_z, axis=1)

        return lidar[lidar_valid_xyz], boxes3d[box_valid_xyz > 0]

    return lidar[lidar_valid_xyz]


def snap_labels(lidar, labels):
    '''
    Compress the labels to eliminate blank space between the last
    point in every dimension, and the bounding box (especially Z)

    Parameters:
        lidar (np.ndarray): (N, 3) point cloud
        labels (np.ndarray): (X, 8, 3) labels in corner notation

    Returns:
        labels (np.ndarray): (X, 8, 3) modified input
    '''
    new_labels = []
    add_thresh = 0.01
    for label in labels:
        min_x, max_x = np.min(label[:, 0]), np.max(label[:, 0])
        min_y, max_y = np.min(label[:, 1]), np.max(label[:, 1])
        min_z, max_z = np.min(label[:, 2]), np.max(label[:, 2])

        x_pts = lidar[:, 0]
        y_pts = lidar[:, 1]
        z_pts = lidar[:, 2]

        valid_x = np.where((x_pts >= min_x) & (x_pts <= max_x))[0]
        valid_y = np.where((y_pts >= min_y) & (y_pts <= max_y))[0]
        valid_z = np.where((z_pts >= min_z) & (z_pts <= max_z))[0]

        box_pts = lidar[np.intersect1d(
            valid_z, np.intersect1d(valid_x, valid_y))]

        box_x = box_pts[:, 0]
        box_y = box_pts[:, 1]
        box_z = box_pts[:, 2]

        min_x, max_x = min(box_x) - add_thresh, max(box_x) + add_thresh
        min_y, max_y = min(box_y) - add_thresh, max(box_y) + add_thresh
        min_z, max_z = min(box_z) - add_thresh, max(box_z) + add_thresh

        bounding_box = np.array([
            [min_x, max_y, min_z],
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, max_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
        ])

        new_labels.append(bounding_box)

    return np.array(new_labels)


def load_kitti_calib(calib_file):
    '''
    Retrieve the transforms from the calibration file

    Parameters:
        calib_file (str): calibration file path

    Returns:
        dict: dict containing necessary transforms
    '''
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0_rect = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0_rect': R0_rect.reshape(3, 3),
            'Tr_velo_to_cam': Tr_velo_to_cam.reshape(3, 4)}


def box3d_cam_to_velo(box3d, tr_velo_to_cam, R0_rect):
    '''
    Transform bounding boxes from center to corner notation
    and transform to velodyne frame

    Parameters:
        box3d (arr): the bouning box in center notation
        Tr (arr): the transform from camera to velodyne

    Returns:
        arr: bounding box in corner notation
    '''

    def camera_to_lidar_box(coord, tr_velo_to_cam, R0_rect):
        R0_formatted = np.eye(4)
        R0_formatted[:3, :3] = R0_rect
        tr_formatted = np.concatenate(
            [tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        coord = np.matmul(np.linalg.inv(R0_formatted), coord)
        coord = np.matmul(np.linalg.inv(tr_formatted), coord)
        return coord[:3].reshape(1, 3)

    def ry_to_rz(ry):
        rz = -ry - np.pi / 2
        limit_degree = 5
        while rz >= np.pi / 2:
            rz -= np.pi
        while rz < -np.pi / 2:
            rz += np.pi

        # So we don't have -pi/2 and pi/2
        if abs(rz + np.pi / 2) < limit_degree / 180 * np.pi:
            rz = np.pi / 2
        return rz

    # KITTI labels are formatted [hwlxyzr]
    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]

    # Position in labels are in cam coordinates. Transform to lidar coords
    cam = np.expand_dims(np.array([tx, ty, tz, 1]), 1)
    translation = camera_to_lidar_box(cam, tr_velo_to_cam, R0_rect)

    rotation = ry_to_rz(ry)

    # Very similar code as in box3d_center_to_corner in conversions.py
    # Create the bounding box outline (to be transposed)
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [0, 0, 0, 0, h, h, h, h]])

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


def load_kitti_label(label_file, tr_velo_to_cam, R0_rect):
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
        box3d_corner = box3d_cam_to_velo(obj[8:], tr_velo_to_cam, R0_rect)

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
