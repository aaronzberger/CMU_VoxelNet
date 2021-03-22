'''
For each point cloud, labels consist of 3d bounding boxes of objects with
little intra-class shape variation
(all tree trunks in a point cloud are of similar width and point density,
bushes are of similar point density, etc)

We want to avoid augmenting data in any way that would suppress a valid,
learnable trait of the target classes

For example, we should not:
    - Move individual objects from their locations: many objects like trunks
        have learnable dependencies to other objects with specific traits
    - Rotate the point cloud or rotate individual objects:
        it is useful information that trees always stem from the same xy plane,
        and that the plane is always flat on the grid.
        This extends to other objects as well


Proposed Algorithm:

To maintain maximum information within classes across point clouds,
I propose the following augmentations, parts of which are adapted from
https://arxiv.org/pdf/2007.13373.pdf :

Given N objects in class C, voxelize based on class C's voxelization scheme:
    - tree trunks are voxelized in Z only, giving vertically stacked
      partitions, since the shape of the cylinder is repeated along the Z axis
    - bushes and other objects are voxelized in X, Y, and Z, since the shape
      of those objects is independent of axis

For each class C, choose 2 objects in class C, X and Y, A random times,
where A is randomly sample from [Amin, Amax] (Amax may be greater than the number of labels):

    Choose V unique random partitions to swap, where V is randomly sampled
    from [Vmin, Vmax].

    Perform the following operations on the voxel before insertion:
    -Resize
        the voxel to match the target voxel's shape
        by moving points towards or away from the centroid
    -Rotate
        all points in the voxel based on the object's rotation scheme:
            for trees, rotate only within [-20, 20] and [160, 200] degrees
            for other objects, rotate [0, 360] degrees

    Exception: for trunks, only swap the bottom voxels with other bottom voxels,
    to maintain valid, learnable information about tree trunk interactions with other objects

Once voxelized, choose N random voxels sampled from all voxels in all objects, where N is
randomly sampled from [Nmin, Nmax]. For each voxel, add Guassian noise with a constant
standard deviation SD.
'''

from math import ceil
import sys
from random import random, uniform, randint, sample

import numpy as np
from numpy.core.defchararray import center
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2 as cv

from conversions import load_custom_label, box3d_center_to_corner
from viz_3d import visualize_lines_3d
from utils import load_config, snap_labels


####################################################
# Number of swaps
Amin = 10
Amax = 100
# Number of voxels to swap (per object)
Vmin = 2
Vmax = 10
# Max rotation for trunks
max_rotation = np.pi / 8
# Number of noise voxels (per object)
Nmin = 3
Nmax = 10
# STD for adding noise
noise_std = 0.04
# The voxel at which the bottom of the tree turns into the rest
trunk_bottom_end = 5
# After how many additional points to remove or add random points when swapping
num_more_pts_thresh = 10
num_fewer_pts_thresh = 10
num_boxes_away = 3
# Separation between voxels for better visualization (no overlap)
viz_voxel_separation = 0.01
####################################################


def voxelize(lidar, boxes):
    '''
    Convert an input point cloud into a voxelized point cloud
    This method is similar to the one in dataset.py

    Parameters:
        lidar (arr): point cloud
        boxes (arr): truth bounding boxes in corner notation

    Returns:
        arr: (N, H, X, 3): voxels of the bounding boxes,
            where N is the number of bounding boxes
            H is the variable number of the voxels in the bounding box
            X is the variable number of points in the voxel
            and 3 represents [X, Y, Z]
        arr: (N, H, 7): bounds of the voxels in the first array:
            where N and H are the same as above
            and 7 represents mins [X, Y, Z], maxes [X, Y, Z], and label indicator
    '''
    config = load_config(config_name='config_trunk')

    # Shuffle the points
    np.random.shuffle(lidar)

    #     2―――――1       2―――――1
    #    / top /|      /| bck |     -> which points the elements of the input
    #   3―――――0 |     3 |     |        array refer to in the bounding box,
    #   |     | 5     | 6―――――5        by index
    #   |     |/      |/ bot /
    #   7―――――4       7―――――4

    z_change = config['voxel_size']['H']

    ground_truth_voxels = []
    ground_truth_coords = []

    only_label_cloud_ind = np.array([])

    x_pts = lidar[:, 0]
    y_pts = lidar[:, 1]
    z_pts = lidar[:, 2]

    for indicator, box in enumerate(boxes):
        object_voxels = []
        object_coords = []

        z_delta = abs(box[0][2] - box[4][2])
        min_x = min(box[4][0], box[5][0], box[6][0], box[7][0])
        max_x = max(box[4][0], box[5][0], box[6][0], box[7][0])
        min_y = min(box[4][1], box[5][1], box[6][1], box[7][1])
        max_y = max(box[4][1], box[5][1], box[6][1], box[7][1])

        valid_x = np.where((x_pts >= min_x) & (x_pts < max_x))[0]
        valid_y = np.where((y_pts >= min_y) & (y_pts < max_y))[0]
        valid_xy = np.intersect1d(valid_x, valid_y)

        z_bottom = min(box[0][2], box[4][2])

        for _ in range(1, ceil(z_delta / z_change)):
            z_top = z_bottom + z_change

            valid_z = np.where((z_pts >= z_bottom) & (z_pts < z_top))[0]

            valid_xyz = np.intersect1d(valid_xy, valid_z)

            pts = lidar[valid_xyz]

            only_label_cloud_ind = np.unique(
                np.concatenate((only_label_cloud_ind, valid_xyz)))

            object_voxels.append(np.array(pts))
            object_coords.append(np.array([min_x, min_y, z_bottom, max_x, max_y, z_top, 0, indicator]))

            z_bottom += z_change

        ground_truth_voxels.append(object_voxels)
        ground_truth_coords.append(object_coords)

    return np.array(ground_truth_voxels, dtype=object), \
        np.array(ground_truth_coords, dtype=object), \
        np.array(only_label_cloud_ind, dtype=object)


def clip_voxel(voxel, coord, origin_centered=False):
    '''
    Eliminate points outside of the bounds of the voxel

    Parameters:
        voxel (np.ndarray): (X, 3) the points
        coord (np.ndarray): (8) voxel info
            min [X, Y, Z], max [X, Y, Z], yaw, label index
    
    Returns:
        np.ndarray: voxel (modified input)
    '''
    deltas, _ = get_voxel_info(coord)
    boundaries = coord[:6]
    if origin_centered:
        boundaries = [-deltas[0] / 2, -deltas[1] / 2, -deltas[2] / 2,
                      deltas[0] / 2, deltas[1] / 2, deltas[2] / 2]

    x_pts = voxel[:, 0]
    y_pts = voxel[:, 1]
    z_pts = voxel[:, 2]
    valid_x = np.where((x_pts >= boundaries[0])
                       & (x_pts <= boundaries[3]))[0]
    valid_y = np.where((y_pts >= boundaries[1])
                       & (y_pts <= boundaries[4]))[0]
    valid_z = np.where((z_pts >= boundaries[2])
                       & (z_pts <= boundaries[5]))[0]

    # Combine the index arrays and take only valid points
    valid_xyz = np.intersect1d(valid_z, np.intersect1d(valid_x, valid_y))

    return voxel[valid_xyz]


def add_noise(voxel, coord):
    '''
    Add 3D Gaussian noise to a voxel

    Parameters:
        voxel (np.ndarray): (X, 3) the points
        coord (np.ndarray): (8) voxel info
            min [X, Y, Z], max [X, Y, Z], yaw, label index
    
    Returns:
        np.ndarray: voxel (modified input)
    '''
    def random_addition():
        return np.random.normal(loc=0.0, scale=noise_std)

    for point in voxel:
        point[0] += random_addition()
        point[1] += random_addition()
        point[2] += random_addition()
    
    voxel = clip_voxel(voxel, coord, origin_centered=False)

    return voxel


def rotate_voxel(voxel, coord):
    '''
    Rotate a voxel around the yaw axis

    Parameters:
        voxel (np.ndarray): (X, 3) the points
        coord (np.ndarray): (8) voxel info
            min [X, Y, Z], max [X, Y, Z], yaw, label index
        
    Returns:
        np.ndarray: voxel (modified input)
        np.ndarray: coord (modified input)
    '''
    # Decide how many radians to rotate
    # Rotate only close to 0 and 180 deg, or else for rectangles, the XY IoU will decrease
    # and many points will be out of the label
    if randint(0, 1):
        radians = uniform(-max_rotation, max_rotation)
    else:
        radians = uniform(np.pi - max_rotation, np.pi + max_rotation)
    
    # If we are rotating a voxel that was swapped previously,
    # rotate relative to its current yaw
    radians -= coord[6]

    axes = np.array([0, 0, radians])
    rotation = R.from_rotvec(axes)
    voxel = rotation.apply(voxel)
    coord[6] = radians
    voxel = clip_voxel(voxel, coord, origin_centered=True)
    return voxel, coord


def get_voxel_info(coord):
    '''
    Get the information (bounds and centroid) of a voxel

    Parameters:
        coord (np.ndarray): (8)
            min [X, Y, Z], max [X, Y, Z], yaw, label index

    Returns:
        np.ndarray: [X, Y, Z] deltas
        np.ndarray: [X, Y, Z] centroid
    '''

    deltas = np.array([abs(coord[3] - coord[0]),
                       abs(coord[4] - coord[1]),
                       abs(coord[5] - coord[2])])

    centroid = np.mean(
        [coord[3:6], coord[:3]], axis=0)

    return deltas, centroid


def fix_voxel_density(features, coords, label_idx, voxel_idx):
    '''
    Remove or add random points if the specified voxel's point density does
    not match that of its close neighbors

    Parameters:
        voxel (np.ndarray): (X, 3) the points
        coord (np.ndarray): (8) voxel info
            min [X, Y, Z], max [X, Y, Z], yaw, label index
        label_idx (int): index of the object in voxel and coord param
        voxel_idx (int): index of the voxel in the object
            specified by the label_idx param

    Returns:
        np.ndarray: feature
    '''
    # Calculate the local point density around the voxel
    local_num_pts_avg = 0
    num_voxels = 0
    # Voxels above the selected voxel
    for i in range(voxel_idx + 1, min(voxel_idx + num_boxes_away + 1, len(features[label_idx]))):
        local_num_pts_avg += features[label_idx][i].shape[0]
        num_voxels += 1
    # Voxels below the selected voxel
    for i in range(max(voxel_idx - num_boxes_away, 0), voxel_idx):
        local_num_pts_avg += features[label_idx][i].shape[0]
        num_voxels += 1
    local_num_pts_avg /= num_voxels

    num_pts = features[label_idx][voxel_idx].shape[0]
    max_pts = round(local_num_pts_avg + num_more_pts_thresh)
    min_pts = max(round(local_num_pts_avg - num_fewer_pts_thresh), 0)

    if num_pts > max_pts:
        # Keep only the max number of points
        keep_indices = sample(range(0, num_pts), max_pts)
        features[label_idx][voxel_idx] = np.take(features[label_idx][voxel_idx], keep_indices, axis=0)

    deltas, centroid = get_voxel_info(coords[label_idx][voxel_idx])
    boundaries = [-deltas[0] / 2, -deltas[1] / 2, -deltas[2] / 2,
                  deltas[0] / 2, deltas[1] / 2, deltas[2] / 2]

    if num_pts < min_pts:
        # Generate the needed number of points to reach the minimum
        for _ in range(min_pts - num_pts):
            x = max(boundaries[0], min(boundaries[3], np.random.normal(loc=0.0, scale=deltas[0] / 6)))
            y = max(boundaries[1], min(boundaries[4], np.random.normal(loc=0.0, scale=deltas[1] / 6)))
            z = max(boundaries[2], min(boundaries[5], np.random.normal(loc=0.0, scale=deltas[2] / 6)))
            point = np.array([x, y, z]) + centroid
            features[label_idx][voxel_idx] = np.concatenate((features[label_idx][voxel_idx], np.expand_dims(point, axis=0)), axis=0)
    
    return features[label_idx][voxel_idx]


def aug_data(features, coords):
    '''
    Perform data augmentation on the provided object labels:
    swap, resize, rotate, and add noise
    
    Parameters:
        features (np.ndarray): (N, H, X, 3): voxels of the bounding boxes,
            where N is the number of bounding boxes
            H is the variable number of the voxels in the bounding box
            X is the variable number of points in the voxel
            and 3 represents [X, Y, Z]
        coords (np.ndarray): (N, H, 8): bounds of the voxels in the first array:
            where N and H are the same as above
            and 8 represents mins [X, Y, Z], maxes [X, Y, Z], yawm and the label indicator
    
    Returns:
        np.ndarray: features parameter, after data augmentation
        np.ndarray: coords parameter, after data augmentation
        list: coords, but only for the voxels that have been swapped
    '''

    swap_indices = []
    only_swap_coords = []

    for _ in range(randint(Amin, Amax)):
        # Allow label_1 = label_2, for intra-label augmentation
        label_1 = randint(0, features.shape[0] - 1)
        label_2 = randint(0, features.shape[0] - 1)

        swap_indices.append(label_1)
        swap_indices.append(label_2)

        # Calculate the length, width, and height of the voxels in each label (for resizing)
        label_1_deltas, _ = get_voxel_info(coords[label_1][0])
        label_2_deltas, _ = get_voxel_info(coords[label_2][0])

        num_swaps = randint(Vmin, Vmax)

        # Pick V random voxel indices in label_1
        label_1_voxels = sample(range(0, len(features[label_1])), num_swaps)
        # Pick V random voxel indices in label_2, where the chosen index
        # is in the same region of the object as in label_1
        label_2_voxels = []
        for idx in label_1_voxels:
            if idx <= trunk_bottom_end:
                label_2_voxels.append(randint(
                    0,
                    min(trunk_bottom_end, len(features[label_2]) - 1)))
            else:
                label_2_voxels.append(randint(
                    min(trunk_bottom_end, len(features[label_2]) - 1),
                    len(features[label_2]) - 1))
        
        for label_1_voxel_idx, label_2_voxel_idx in zip(
                label_1_voxels, label_2_voxels):

            # Calculate the centroids (average of max and min on each axis)
            _, voxel_1_centroid = get_voxel_info(coords[label_1][label_1_voxel_idx])
            _, voxel_2_centroid = get_voxel_info(coords[label_2][label_2_voxel_idx])

            # Move both voxels so they are centered around the origin
            features[label_1][label_1_voxel_idx] -= voxel_1_centroid
            features[label_2][label_2_voxel_idx] -= voxel_2_centroid

            # Resize the voxels
            features[label_1][label_1_voxel_idx] *= label_2_deltas / label_1_deltas
            features[label_2][label_2_voxel_idx] *= label_1_deltas / label_2_deltas

            # Rotate the voxels
            features[label_1][label_1_voxel_idx], coords[label_2][label_2_voxel_idx] = \
                rotate_voxel(features[label_1][label_1_voxel_idx], coords[label_2][label_2_voxel_idx])
            features[label_2][label_2_voxel_idx], coords[label_1][label_1_voxel_idx] = \
                rotate_voxel(features[label_2][label_2_voxel_idx], coords[label_1][label_1_voxel_idx])

            # Apply the translation so the voxel locations are swapped
            features[label_1][label_1_voxel_idx] += voxel_2_centroid
            features[label_2][label_2_voxel_idx] += voxel_1_centroid

            # Swap the voxels in the features array
            store_label_1_voxel = features[label_1][label_1_voxel_idx]
            features[label_1][label_1_voxel_idx] = features[label_2][label_2_voxel_idx]
            features[label_2][label_2_voxel_idx] = store_label_1_voxel

            features[label_1][label_1_voxel_idx] = fix_voxel_density(
                features, coords, label_1, label_1_voxel_idx)
            features[label_2][label_2_voxel_idx] = fix_voxel_density(
                features, coords, label_2, label_2_voxel_idx)

            # Swap the box indices in the coords array (for coloring)
            store_label_1_color = coords[label_1][label_1_voxel_idx][-1]
            coords[label_1][label_1_voxel_idx][-1] = coords[label_2][label_2_voxel_idx][-1]
            coords[label_2][label_2_voxel_idx][-1] = store_label_1_color


    for idx in swap_indices:
        only_swap_coords.append(coords[idx])

    # Apply noise to some random label voxels
    flattened_coord_indices = []

    for object_idx, object in enumerate(coords):
        for voxel_idx, _ in enumerate(object):
            flattened_coord_indices.append((object_idx, voxel_idx))

    num_voxels = min(randint(Nmin * len(coords), Nmax * len(coords)), len(coords))
    voxel_indices = sample(range(0, len(flattened_coord_indices)), num_voxels)

    for idx in voxel_indices:
        object_idx, voxel_idx = flattened_coord_indices[idx]

        features[object_idx][voxel_idx] = add_noise(
            features[object_idx][voxel_idx], coords[object_idx][voxel_idx])

    return features, coords, only_swap_coords


def display_voxels(coords, cloud, colors):
    '''
    Given the coordinates of voxels, convert to corner notation and display

    Parameters:
        coords (np.ndarray): (H, 8) voxel info
        cloud (np.ndarray): (N, 3) point cloud
        colors (np.ndarray): (X, 3) RGB indicator for each object index
    '''
    config = load_config()
    center_boxes = []
    box_colors = []

    for label in coords:
        for voxel in label:
            centroid = np.mean([voxel[3:6], voxel[:3]], axis=0)

            if voxel[2] % config['voxel_size']['H'] == 0:
                voxel[2] += viz_voxel_separation
                voxel[5] -= viz_voxel_separation

            x, y, z = centroid[:3]
            h = voxel[5] - voxel[2]
            w = voxel[4] - voxel[1]
            l = voxel[3] - voxel[0]
            r = voxel[6]

            center_boxes.append(np.array([x, y, z, h, w, l, r]))
    
            box_colors.append([colors[int(voxel[-1])] for _ in range(12)])
    
    all_voxel_boxes = box3d_center_to_corner(np.array(center_boxes), z_middle=True)


    visualize_lines_3d(pointcloud=cloud, gt_boxes=np.array(all_voxel_boxes),
                       gt_box_colors=box_colors, reduce_pts=True)


def features_to_cloud(features):
    '''
    Convert object-wise, voxel-wise points to a full point cloud list

    Parameters:
        features (np.ndarray): (N, H, 3) features

    Returns:
        np.ndarray: (X, 3): point cloud
    '''
    cloud = np.empty((0, 3))
    for object in features:
        for voxel in object:
            cloud = np.concatenate((cloud, voxel), axis=0)
    return cloud


if __name__ == '__main__':
    if len(sys.argv) == 2:
        cloud_path = '/home/aaron/tree_pcl_data/cloud_{}/cloud_{}.pcd'.format(sys.argv[1], sys.argv[1])
        label_path = '/home/aaron/tree_pcl_data/cloud_{}/cloud_{}_labels.txt'.format(sys.argv[1], sys.argv[1])
    else:
        cloud_path, label_path = sys.argv[1:3]

    config = load_config('config_trunk')
    cloud = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(cloud.points)

    # lidar = filter_pointcloud(points, config='config_trunk')
    lidar = points

    # Get an (X, 8, 3) array of the labels
    labels = load_custom_label(label_path)

    labels = snap_labels(lidar, labels)

    # Voxelize the labels
    features, coords, cloud_ind = voxelize(lidar, labels)

    # Select only the points inside of labels
    cloud_ind = cloud_ind.astype(int)
    only_label_pts = lidar[cloud_ind]

    # Generate a list of colors, one for each object instance (N, 3),
    # where N is the number of labels, and 3 encodes [R, G, B]
    label_colors = []
    for _ in range(features.shape[0]):
        label_colors.append([uniform(0.0, 0.9), uniform(0.0, 0.9), uniform(0.0, 0.9)])

    # Visualize the entire point cloud with black bounding boxes
    black = [0, 0, 0]
    black_boxes = []
    for _ in range(labels.shape[0]):
        black_boxes.append([black for _ in range(12)])

    print('''Displaying the full point cloud and all labels:
        {} labels and {} points'''.format(labels.shape[0], lidar.shape[0]))
    visualize_lines_3d(
        pointcloud=lidar, gt_boxes=labels, gt_box_colors=black_boxes, reduce_pts=False)
    print('''Displaying the full point cloud and all labels compressed:
        {} labels and {} points'''.format(labels.shape[0], lidar.shape[0]))
    visualize_lines_3d(
        pointcloud=lidar, gt_boxes=labels, gt_box_colors=black_boxes, reduce_pts=False)

    # Visualize only points inside the labels and the bounding boxes
    color_boxes = []
    for i in range(labels.shape[0]):
        color_boxes.append([label_colors[i] for _ in range(12)])

    print('''Displaying only points inside labels:
        {} labels and {} points\n'''.format(labels.shape[0], only_label_pts.shape[0]))
    visualize_lines_3d(
        pointcloud=only_label_pts, gt_boxes=labels, gt_box_colors=color_boxes,
        reduce_pts=True)

    # print('Displaying the voxelized labels\n')
    # display_voxels(
    #     coords, cloud=only_label_pts, colors=label_colors)

    # augmented_features, augmented_coords, only_swap_coords = aug_data(features, coords)
    # augmented_cloud = features_to_cloud(augmented_features)

    # print('Displaying augmented data\n')
    # display_voxels(
    #     only_swap_coords, cloud=augmented_cloud, colors=label_colors)
    

    # Focus on one label
    # label_idx = randint(0, len(coords) - 1)
    # deltas = np.array([abs(coords[label_idx][0][3] - coords[label_idx][0][0]),
    #                    abs(coords[label_idx][0][4] - coords[label_idx][0][1]),
    #                    abs(coords[label_idx][len(coords[label_idx]) - 1][5] - coords[label_idx][0][2])])

    # image_dims = int(deltas[2] * 50 + 30), int(deltas[0] * 100 + 30)
    
    # for i in range(20):
    #     augmented_features, augmented_coords, only_swap_coords = aug_data(features, coords)
    #     augmented_cloud = features_to_cloud(augmented_features)

    #     image = np.zeros((image_dims[1], image_dims[0], 3), dtype=np.uint8)

    #     for voxel in augmented_features[label_idx]:
    #         for point in voxel:
    #             y_image = int((point[0] - coords[label_idx][0][0]) * 100 + 15)
    #             x_image = int((point[2]) * 50 + 15)
    #             image = cv.circle(
    #                 image, (x_image, y_image), radius=1, color=(0, 0, 255), thickness=-1)
        
    #     cv.imwrite('/home/aaron/CMU_VoxelNet/viz/data_aug/obj{}_iter{}.jpg'.format(label_idx, i), image)

        

