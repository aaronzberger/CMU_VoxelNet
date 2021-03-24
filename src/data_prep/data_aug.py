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

import copy
from math import ceil
import os
import sys
from random import uniform, randint, sample

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation as R

from config import data_dir
from conversions import box3d_center_to_corner, load_numpy_label
from utils import load_config, snap_labels
from viz_3d import visualize_lines_3d


####################################################
# Number of swaps (objects)
Amin = 10
Amax = 150
# Number of voxels to swap (per object)
Vmin = 2
Vmax = 10
# Max rotation for trunks
max_rotation = np.pi / 8
# Number of noise voxels (per object)
Nmin = 3
Nmax = 12
# STD for adding noise to label points
label_noise_std = 0.04
# STD for adding noise to non-label points
cloud_noise_std = 0.1
# The voxel at which the bottom of the tree turns into the rest
trunk_bottom_end = 5
# After how many additional points to remove or add random points when swapping
num_more_pts_thresh = 10
num_fewer_pts_thresh = 10
num_boxes_away = 3
# Separation between voxels for better visualization (no overlap)
viz_voxel_separation = 0.01
# Number of objects to remove
Rmin = 3
Rmax = 15
# Min percentage of non-label points to remove
RPmin = 0.02
# Max percentage of non-label points to remove
RPmax = 0.2
####################################################


def voxelize(lidar, boxes):
    '''
    Convert an input point cloud into a voxelized point cloud
    This method is similar to the one in dataset.py

    Parameters:
        lidar (arr): point cloud
        boxes (arr): truth bounding boxes in corner notation

    Returns:
        np.ndarray: (N, H, X, 3): points in the voxels,
            where N is the number of bounding boxes
            H is the variable number of the voxels in the bounding box
            X is the variable number of points in the voxel
            and 3 represents [X, Y, Z]
        np.ndarray: (N, H, 8): bounds of the voxels:
            where N and H are the same as above
            and 8 represents
                [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]
    '''
    config = load_config()

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
            object_coords.append(np.array(
                [min_x, min_y, z_bottom, max_x, max_y, z_top, 0, indicator]))

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
            [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]

    Returns:
        np.ndarray: voxel (modified input)
    '''
    ranges, _ = get_voxel_info(coord)
    boundaries = coord[:6]
    if origin_centered:
        boundaries = [-ranges[0] / 2, -ranges[1] / 2, -ranges[2] / 2,
                      ranges[0] / 2, ranges[1] / 2, ranges[2] / 2]

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


def trim_pointcloud(pointcloud):
    '''
    Trim the pointcloud after data augmentation

    Parameters:
        pointcloud (np.ndarray): (N, 3) points

    Returns:
        np.ndarray: (N, 3): filtered input
    '''
    config = load_config()

    pointcloud[:, 2] = np.maximum(
        pointcloud[:, 2], np.zeros_like(pointcloud[:, 2]))

    valid_x = np.where((pointcloud[:, 0] >= config['pcl_range']['X1'])
                       & (pointcloud[:, 0] <= config['pcl_range']['X2']))[0]
    valid_y = np.where((pointcloud[:, 1] >= config['pcl_range']['Y1'])
                       & (pointcloud[:, 1] <= config['pcl_range']['Y2']))[0]
    valid_z = np.where((pointcloud[:, 2] >= config['pcl_range']['Z1'])
                       & (pointcloud[:, 2] <= config['pcl_range']['Z2']))[0]

    valid_xyz = np.intersect1d(valid_z, np.intersect1d(valid_x, valid_y))

    return pointcloud[valid_xyz]


def add_noise(voxel, std, clip=True, coord=None):
    '''
    Add 3D Gaussian noise to a voxel

    Parameters:
        voxel (np.ndarray): (X, 3) the points
        std (float): standard deviation of noise
        clip (bool): whether to clip the voxel according to its
            coordinates after adding noise
        coord (np.ndarray): (8) voxel info
            [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]

    Returns:
        np.ndarray: voxel (modified input)
    '''
    if clip is True and coord is None:
        raise ValueError('If clip is True, coord argument must be provided')

    def random_addition():
        return np.random.normal(loc=0.0, scale=std)

    for point in voxel:
        point[0] += random_addition()
        point[1] += random_addition()
        point[2] += random_addition()

    if clip:
        voxel = clip_voxel(voxel, coord, origin_centered=False)

    return voxel


def rotate_voxel(voxel, coord):
    '''
    Rotate a voxel around the yaw axis

    Parameters:
        voxel (np.ndarray): (X, 3) the points
        coord (np.ndarray): (8) voxel info
            [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]

    Returns:
        np.ndarray: voxel (modified input)
        np.ndarray: coord (modified input)
    '''
    # Decide how many radians to rotate (Rotate only close to 0 and 180 deg
    # or else for rectangles, the XY IoU will decrease)
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
            [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]

    Returns:
        np.ndarray: [X, Y, Z] ranges, or deltas
        np.ndarray: [X, Y, Z] centroid
    '''

    ranges = np.array([abs(coord[3] - coord[0]),
                       abs(coord[4] - coord[1]),
                       abs(coord[5] - coord[2])])

    centroid = np.mean(
        [coord[3:6], coord[:3]], axis=0)

    return ranges, centroid


def fix_voxel_density(features, coords, label_idx, voxel_idx):
    '''
    Remove or add random points if the specified voxel's point density does
    not match that of its close neighbors

    Parameters:
        voxel (np.ndarray): (X, 3) the points
        coord (np.ndarray): (8) voxel info
            [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]
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
    for i in range(voxel_idx + 1, min(
            voxel_idx + num_boxes_away + 1, len(features[label_idx]))):
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
        features[label_idx][voxel_idx] = np.take(
            features[label_idx][voxel_idx], keep_indices, axis=0)

    ranges, centroid = get_voxel_info(coords[label_idx][voxel_idx])
    boundaries = [-ranges[0] / 2, -ranges[1] / 2, -ranges[2] / 2,
                  ranges[0] / 2, ranges[1] / 2, ranges[2] / 2]

    if num_pts < min_pts:
        # Generate the needed number of points to reach the minimum
        for _ in range(min_pts - num_pts):
            x = max(boundaries[0], min(boundaries[3],
                    np.random.normal(loc=0.0, scale=ranges[0] / 6)))
            y = max(boundaries[1], min(boundaries[4],
                    np.random.normal(loc=0.0, scale=ranges[1] / 6)))
            z = max(boundaries[2], min(boundaries[5],
                    np.random.normal(loc=0.0, scale=ranges[2] / 6)))

            point = np.array([x, y, z]) + centroid
            features[label_idx][voxel_idx] = np.concatenate(
                (features[label_idx][voxel_idx],
                 np.expand_dims(point, axis=0)), axis=0)

    return features[label_idx][voxel_idx]


def aug_data(features, coords):
    '''
    Perform data augmentation on the provided object labels:
        swap, resize, rotate, adjust density

    Parameters:
        features (np.ndarray): (N, H, X, 3): points in the voxels,
            where N is the number of bounding boxes
            H is the variable number of the voxels in the bounding box
            X is the variable number of points in the voxel
            and 3 represents [X, Y, Z]
        coords (np.ndarray): (N, H, 8): bounds of the voxels:
            where N and H are the same as above
            and 8 represents
                [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]

    Returns:
        np.ndarray: features parameter, after data augmentation
        np.ndarray: coords parameter, after data augmentation
        list: coords, but only for the voxels that have been swapped
    '''
    # Deep copy the arrays or else the outputs will be references to the inputs
    features = copy.deepcopy(features)
    coords = copy.deepcopy(coords)

    swap_indices = []
    only_swap_coords = []

    for _ in range(randint(Amin, Amax)):
        # Allow label_1 = label_2, for intra-label augmentation
        label_1 = randint(0, features.shape[0] - 1)
        label_2 = randint(0, features.shape[0] - 1)

        swap_indices.append(label_1)
        swap_indices.append(label_2)

        # Get the label ranges for each label
        label_1_ranges, _ = get_voxel_info(coords[label_1][0])
        label_2_ranges, _ = get_voxel_info(coords[label_2][0])

        num_swaps = randint(Vmin, Vmax)

        # Pick V random voxel indices in label_1
        # Always leave the bottom voxel unchanged as a reference for the yaw
        label_1_voxels = sample(
            range(1, len(features[label_1])),
            min(len(features[label_1]) - 1, num_swaps))

        # Pick V random voxel indices in label_2, where the chosen index
        # is in the same region of the object as in label_1
        label_2_voxels = []
        for idx in label_1_voxels:
            if idx <= trunk_bottom_end:
                label_2_voxels.append(randint(
                    1,
                    min(trunk_bottom_end, len(features[label_2]) - 1)))
            else:
                label_2_voxels.append(randint(
                    min(trunk_bottom_end, len(features[label_2]) - 1),
                    len(features[label_2]) - 1))

        for label_1_voxel_idx, label_2_voxel_idx in zip(
                label_1_voxels, label_2_voxels):
            # Get the centroids of the voxels
            _, voxel_1_centroid = get_voxel_info(
                coords[label_1][label_1_voxel_idx])
            _, voxel_2_centroid = get_voxel_info(
                coords[label_2][label_2_voxel_idx])

            # Move both voxels so they are centered around the origin
            features[label_1][label_1_voxel_idx] -= voxel_1_centroid
            features[label_2][label_2_voxel_idx] -= voxel_2_centroid

            # Resize the voxels
            features[label_1][label_1_voxel_idx] *= \
                label_2_ranges / label_1_ranges
            features[label_2][label_2_voxel_idx] *= \
                label_1_ranges / label_2_ranges

            # Rotate the voxels (randomly)
            features[label_1][label_1_voxel_idx], \
                coords[label_2][label_2_voxel_idx] = \
                rotate_voxel(features[label_1][label_1_voxel_idx],
                             coords[label_2][label_2_voxel_idx])
            features[label_2][label_2_voxel_idx], \
                coords[label_1][label_1_voxel_idx] = \
                rotate_voxel(features[label_2][label_2_voxel_idx],
                             coords[label_1][label_1_voxel_idx])

            # Apply the translation so the voxels are swapped w.r.t the cloud
            features[label_1][label_1_voxel_idx] += voxel_2_centroid
            features[label_2][label_2_voxel_idx] += voxel_1_centroid

            # Swap the voxel arrays within the features array
            store_label_1_voxel = features[label_1][label_1_voxel_idx]
            features[label_1][label_1_voxel_idx] = \
                features[label_2][label_2_voxel_idx]
            features[label_2][label_2_voxel_idx] = store_label_1_voxel

            # Ensure the voxel densities are close to those of near voxels
            features[label_1][label_1_voxel_idx] = fix_voxel_density(
                features, coords, label_1, label_1_voxel_idx)
            features[label_2][label_2_voxel_idx] = fix_voxel_density(
                features, coords, label_2, label_2_voxel_idx)

            # Swap the box indices in the coords array (for coloring)
            store_label_1_color = coords[label_1][label_1_voxel_idx][-1]
            coords[label_1][label_1_voxel_idx][-1] = \
                coords[label_2][label_2_voxel_idx][-1]
            coords[label_2][label_2_voxel_idx][-1] = store_label_1_color

    for idx in swap_indices:
        only_swap_coords.append(coords[idx])

    return features, coords, only_swap_coords


def aug_cloud(features, coords, non_label_pts):
    '''
    Apply data augmentation on the point cloud as a whole:
        add noise, remove random labels, remove random points,
        rotate and translate

    Parameters:
        features (np.ndarray): (N, H, X, 3): points in the voxels,
            where N is the number of bounding boxes
            H is the variable number of the voxels in the bounding box
            X is the variable number of points in the voxel
            and 3 represents [X, Y, Z]
        coords (np.ndarray): (N, H, 8): bounds of the voxels:
            where N and H are the same as above
            and 8 represents
                [Xₘᵢₙ, Yₘᵢₙ, Zₘᵢₙ, Xₘₐₓ, Yₘₐₓ, Zₘₐₓ, θ, label_index]
        non_label_pts (np.ndarray): (X, 3) points outside the labels

    Returns:
        np.ndarray: modified features parameter
        np.ndarray: modified coords parameter
        np.ndarray: new entire pointcloud (labels and otherwise)
    '''
    # Deep copy the arrays or else the outputs will be references to the inputs
    features = copy.deepcopy(features)
    coords = copy.deepcopy(coords)

    # So we can randomly pick any voxel in any object with unform distribution,
    # match every index with an index in a flattened array
    flattened_coord_indices = []

    for object_idx, object in enumerate(coords):
        for voxel_idx, _ in enumerate(object):
            flattened_coord_indices.append((object_idx, voxel_idx))

    num_voxels = min(randint(Nmin * len(coords), Nmax * len(coords)),
                     len(coords))
    voxel_indices = sample(range(0, len(flattened_coord_indices)), num_voxels)

    # Apply noise to some random label voxels
    for idx in voxel_indices:
        object_idx, voxel_idx = flattened_coord_indices[idx]

        features[object_idx][voxel_idx] = add_noise(
            features[object_idx][voxel_idx], std=label_noise_std, clip=True,
            coord=coords[object_idx][voxel_idx])

    # Select indices of labels to keep
    remove_indices = sample(range(0, len(flattened_coord_indices)),
                            randint(min(Rmin, len(flattened_coord_indices)),
                                    min(Rmax, len(flattened_coord_indices))))

    for i, idx in enumerate(remove_indices):
        object_idx, voxel_idx = flattened_coord_indices[idx]
        object_idx -= i

        features = np.delete(features, object_idx, axis=0)
        coords = np.delete(coords, object_idx, axis=0)

    # Add random noise to the non_label points
    non_label_pts = add_noise(non_label_pts, std=cloud_noise_std, clip=False)

    # Remove some random points from the non-label points
    cloud_keep_indices = sample(
        range(0, non_label_pts.shape[0]),
        non_label_pts.shape[0] -
        randint(min(int(RPmin * non_label_pts.shape[0]),
                    non_label_pts.shape[0]),
                min(int(RPmax * non_label_pts.shape[0]),
                    non_label_pts.shape[0])))
    non_label_pts = non_label_pts[cloud_keep_indices]

    new_pointcloud = np.concatenate(
        (features_to_cloud(features), non_label_pts), axis=0)

    # Rotate the entire pointcloud
    radians = uniform(-np.pi, np.pi)

    axes = np.array([0, 0, radians])
    rotation = R.from_rotvec(axes)
    new_pointcloud = rotation.apply(new_pointcloud)

    # Translate the pointcloud
    x_translate, y_translate = uniform(-10, 10), uniform(-10, 10)
    new_pointcloud[:, 0] += x_translate
    new_pointcloud[:, 1] += y_translate

    # Crop the pointcloud
    new_pointcloud = trim_pointcloud(new_pointcloud)

    # Rotate the labels
    for object in coords:
        for voxel in object:
            ranges, centroid = get_voxel_info(voxel)
            centroid = rotation.apply(centroid)
            centroid[0] += x_translate
            centroid[1] += y_translate
            ranges = np.divide(ranges, 2)
            coord = [centroid[0] - ranges[0], centroid[1] - ranges[1],
                     centroid[2] - ranges[2], centroid[0] + ranges[0],
                     centroid[1] + ranges[1], centroid[2] + ranges[2]]
            voxel[:6] = coord
            voxel[6] += radians

    return features, coords, new_pointcloud


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

    all_voxel_boxes = box3d_center_to_corner(
        np.array(center_boxes), z_middle=True)

    visualize_lines_3d(pointcloud=cloud, gt_boxes=np.array(all_voxel_boxes),
                       gt_box_colors=box_colors, reduce_pts=False)


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


def coords_to_center(coords):
    ''''''
    output_labels = []

    for object in coords:
        voxel = object[0]
        voxel[5] = object[-1][5]
        ranges, centroid = get_voxel_info(voxel)

        output_labels.append(np.array(['trunk',
                                       centroid[0], centroid[1], centroid[2],
                                       ranges[0], ranges[1], ranges[2],
                                       voxel[6]]))

    return np.array(output_labels)


if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        raise ValueError(
            'Usage: data_aug [PCL NUMBER] [NUMBER OF AUGMENTATIONS]' +
            ' Add \'viz\' at the end to visualize each augmentation')

    visualize = False
    if len(sys.argv) == 4 and sys.argv[3] == 'viz':
        visualize = True

    # Load the labeled pointcloud and labels
    data = np.load(
        data_dir + '/cropped/cloud_{}.npz'.format(sys.argv[1]))
    pointcloud, labels = data['pointcloud'], data['labels']

    # Convert input labels to corner notation and snap them to fit
    labels = load_numpy_label(labels)

    try:
        labels = snap_labels(pointcloud, labels)
    except ValueError:
        print('At least one label contained no points:' +
              'the labels were probably saved incorrectly')
        sys.exit(2)

    # Voxelize the labels
    features, coords, cloud_ind = voxelize(pointcloud, labels)

    # Separate the points by whether they are in a label
    cloud_ind = cloud_ind.astype(int)
    only_label_pts = pointcloud[cloud_ind]
    non_label_pts = np.delete(pointcloud, cloud_ind, axis=0)

    # Generate a list of colors, one for each object instance (N, 3),
    # where N is the number of labels, and 3 encodes [R, G, B]
    label_colors = []
    for _ in range(features.shape[0]):
        label_colors.append(
            [uniform(0.0, 0.9), uniform(0.0, 0.9), uniform(0.0, 0.9)])

    color_boxes = []
    for i in range(labels.shape[0]):
        color_boxes.append([label_colors[i] for _ in range(12)])

    for i in range(int(sys.argv[2])):
        if visualize:
            print('Displaying original...')
            display_voxels(
                coords=coords, cloud=pointcloud, colors=label_colors)
        augmented_features, augmented_coords, only_swap_coords = \
            aug_data(features, coords)
        augmented_cloud = features_to_cloud(augmented_features)
        if visualize:
            print('Displaying augmented labels for index {}...'.format(i))
            display_voxels(
                coords=augmented_coords, cloud=augmented_cloud,
                colors=label_colors)
        augmented_features, augmented_coords, new_pointcloud = \
            aug_cloud(augmented_features, augmented_coords, non_label_pts)
        if visualize:
            print('Displaying final point cloud for index {}...'.format(i))
            display_voxels(
                coords=augmented_coords, cloud=new_pointcloud,
                colors=label_colors)

        save_path = data_dir + '/cropped/cloud_{}_iter_{}'.format(
            sys.argv[1], i)
        np.savez(save_path, pointcloud=new_pointcloud,
                 labels=coords_to_center(augmented_coords))
        print('Saved index {} augmentation of cloud {} to {}'.format(
            i, sys.argv[1], save_path))
