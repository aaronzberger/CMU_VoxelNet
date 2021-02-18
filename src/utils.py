from __future__ import division
import os
import json
import math
import errno
import time

import torch
import numpy as np

from config import base_dir


def load_config():
    '''
    Load the configuration file

    Returns:
        config (dict): Python dictionary of hyperparameter name-value pairs
    '''

    path = os.path.join(base_dir, 'config.json')

    with open(path) as file:
        config = json.load(file)

    return config


def get_model_path(name):
    '''
    Get the path to save the state_dict

    Parameters:
        name (any): name of the epoch (can be a number or a token)
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


def get_filtered_lidar(lidar, boxes3d=None):
    '''
    Crop a lidar pointcloud to the dimensions specified in config

    Parameters:
        lidar (arr): the point cloud
        boxes3d (arr): Ground truth boxes to crop

    Returns:
        arr: cropped point cloud
        arr: cropped ground truth boxes
    '''
    config = load_config()

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


def box3d_center_to_corner_batch(boxes_center):
    '''
    Transform bounding boxes from center to corner notation

    Parameters:
        boxes_center (arr): (N, X, 7):
            boxes in center notation

    Returns:
        arr: bounding box in corner notation
    '''
    boxes_center = boxes_center.cpu().numpy()
    batch_size = boxes_center.shape[0]
    num_boxes = boxes_center.shape[1]

    # To return
    corner_boxes = np.zeros((batch_size, num_boxes, 8, 3))

    for batch_id in range(batch_size):
        boxes = boxes_center[batch_id]

        for box_num, box in enumerate(boxes):
            translation = box[0:3]
            size = box[3:6]
            rotation = [0, 0, box[-1]]

            h, w, l = size[0], size[1], size[2]
            bounding_box = np.array([
                [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                [0, 0, 0, 0, h, h, h, h]])

            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]])

            # Just repeat [x, y, z] eight times
            eight_points = np.tile(translation, (8, 1))

            # Add rotated bounding box to the center position to obtain corners
            cornerPosInVelo = np.dot(
                rotation_matrix, bounding_box) + eight_points.transpose()
            box3d = cornerPosInVelo.transpose()

            corner_boxes[batch_id][box_num] = box3d

    return corner_boxes


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
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    bounding_box = np.array([[-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                             [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                             [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotation_matrix = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    eight_points = np.tile(t_lidar, (8, 1))

    cornerPosInVelo = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)


def anchors_center_to_corner(anchors):
    '''
    Convert anchors to corner notation (BEV only)
    (N, 7) -> (N, 4, 3)

    Parameters:
        anchors (arr): the anchors in the form [xyzhwlr]

    Returns:
        arr: the anchors in corner notation
    '''
    N = anchors.shape[0]

    anchors_corner = np.zeros((N, 4, 2))

    for i in range(N):
        anchor = anchors[i]
        h, w, l = anchor[3:6]
        rz = anchor[-1]

        bounding_box = np.array([
            [-l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2]])

        # re-create 3D bounding box in velodyne frame
        rotation_matrix = np.array([
            [np.cos(rz), -np.sin(rz)],
            [np.sin(rz), np.cos(rz)]])

        four_points = np.tile(anchor[:2], (4, 1))

        # Add rotation matrix to (2, 4) of Xs, then Ys
        cornerPosInVelo = np.dot(rotation_matrix, bounding_box) + \
            four_points.transpose()

        # [XXXX,YYYY] -> [XY,XY,XY,XY]
        box2d = cornerPosInVelo.transpose()
        anchors_corner[i] = box2d

    return anchors_corner


def corner_to_standup_box2d_batch(boxes_corner):
    '''
    Convert corner coordinates to xyxy form boxes
    (N, 4, 2) → (N, 4): [x1, y1, x2, y2]

    Parameters:
        boxes_corner (arr): the boxes in corner notation

    Returns:
        arr: the boxes with the max and min x and y values from the input
    '''
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))

    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)  # X1
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)  # Y1
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)  # X2
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)  # Y2

    return standup_boxes2d


def box3d_corner_to_center_batch(box3d_corner):
    '''
    Convert a batch of ground truth bounding boxes from corner to center
    notation (N, 8, 3) -> (N, 7)

    Parameters:
        box3d_corner (arr): bounding boxes in corner notation

    Returns:
        arr: bounding boxes in xyzhwlr notation
    '''
    assert box3d_corner.ndim == 3

    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    #     2―――――1       2―――――1
    #    / top /|      /|     |     -> which points the elements of the input
    #   3―――――0 |     3 |     |        array refer to in the bounding box,
    #   |     | 5     | 6―――――5        by index
    #   |     |/      |/ bot /
    #   7―――――4       7―――――4

    # Define height as the difference between the top and bottom coordinates
    height = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2],
                         axis=1, keepdims=True))
    #                      _______________
    # width = average of  √ (pt₁ - pt₂)²    for all points on the same side
    width = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4
    #                       _______________
    # length = average of  √ (pt₁ - pt₂)²   for all points on the opposite side
    length = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
              np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
              np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
              np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    # theta = average of atan2(Δy, Δx) for non-diagonals on the top of the bbox
    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, height, width, length, theta], axis=1).reshape(
        box3d_corner.shape[0], 7)


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


def delta_to_boxes3d(deltas, anchors):
    '''
    Convert regression map deltas to bounding boxes

    Parameters:
        deltas (arr): (N, W, L, 14): the regression output,
            where N = batch size
        anchors (arr): (X, 7): the anchors, where X = number of anchors

    ReturnsL
        arr: the bounding boxes
    '''
    N = deltas.shape[0]
    deltas = deltas.view(N, -1, 7)
    anchors = torch.FloatTensor(anchors)
    boxes3d = torch.zeros_like(deltas)

    if deltas.is_cuda:
        anchors = anchors.cuda()
        boxes3d = boxes3d.cuda()

    anchors_reshaped = anchors.view(-1, 7)
    #       _______________
    # dᵃ = √ (lᵃ)² + (wᵃ)²      is the diagonal of the base
    #                           of the anchor box (See 2.2)
    anchors_diagonal = torch.sqrt(
        anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)

    # Repeat so we can multiply X and Y in same mul operation
    anchors_diagonal = anchors_diagonal.repeat(N, 2, 1).transpose(1, 2)

    # Copy over the batch size
    anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

    # Δx = (xᵍ - xᵃ) / dᵃ and Δy = (yᵍ - yᵃ) / dᵃ
    # so x = (Δx * dᵃ) + xᵃ and y = (Δy * dᵃ) + yᵃ
    boxes3d[..., [0, 1]] = torch.mul(
        deltas[..., [0, 1]], anchors_diagonal) \
        + anchors_reshaped[..., [0, 1]]

    # Δz = (zᵍ - zᵃ) / hᵃ so z = (Δz * hᵃ) + zᵃ
    boxes3d[..., [2]] = torch.mul(
        deltas[..., [2]], anchors_reshaped[..., [3]]) \
        + anchors_reshaped[..., [2]]

    # Δw = log(wᵍ / wᵃ) so w = e^(Δw) * wᵃ
    boxes3d[..., [3, 4, 5]] = torch.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

    # Δ𝜃 = 𝜃ᵍ - 𝜃ᵃ, so 𝜃 = Δ𝜃 + 𝜃ᵃ
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d


def draw_boxes(reg_map, prob_score_map, calibs):
    config = load_config()

    # View as list of anchors
    prob_score_map = prob_score_map.view(config['batch_size'], -1)

    # Convert regression map deltas to actual bounding boxes
    batch_boxes3d = delta_to_boxes3d(reg_map, get_anchors())

    # Only use predictions where the prediction was > threshold
    mask = torch.gt(torch.sigmoid(prob_score_map),
                    config['nms_score_threshold'])

    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

    nonzero = torch.nonzero(mask).shape[0]
    return_boxes = torch.zeros((config['batch_size'], nonzero, 7))
    return_scores = torch.zeros((config['batch_size'], nonzero))

    for batch_id in range(config['batch_size']):
        boxes3d = torch.masked_select(
            batch_boxes3d[batch_id], mask_reg[batch_id]).view(-1, 7)
        scores = torch.masked_select(
            prob_score_map[batch_id], mask[batch_id])

        return_boxes[batch_id] = boxes3d
        return_scores[batch_id] = scores

    # Convert the final boxes from center [xyzhwlr] to corner points notation
    boxes_corner = box3d_center_to_corner_batch(return_boxes, calibs)

    return boxes_corner, return_boxes, return_scores


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
