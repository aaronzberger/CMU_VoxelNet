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

Given N objects in class C, voxelize based on class C's voxelization protocol:
    - tree trunks are voxelized in Z only, giving vertically stacked
      partitions, since the shape of the cylinder is repeated along the Z axis
    - bushes and other objects are voxelized in X, Y, and Z, since the shape
      of those objects is independent of axis

For each class C, choose 2 objects in class C, X and Y, A random times,
where A is randomly sample from [Amin, Amax]

Choose V unique random partitions to swap, where V is randomly sampled
from [Vmin, Vmax].

Perform the following operations on the voxel before insertion:
-Resize
    the voxel by moving points towards or away from the centroid,
-Rotate
    the entire voxel around the yaw axis a random theta [0, 180deg]
-Add noise
    to all points in the voxel using Guassian distribution with a constant sd

Exception: for the bottom-most voxel and the top-most voxel in a tree trunk,
only swap with other bottom-most and top-most voxels respectively, to maintain
valid, learnable information about a tree trunks interaction with other objects
'''

from math import ceil
import sys
import numpy as np
from random import uniform, randint, sample
from utils import load_config, filter_pointcloud
from conversions import load_custom_label
import open3d as o3d
from viz_3d import visualize_lines_3d


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
            object_coords.append(np.array([min_x, min_y, z_bottom, max_x, max_y, z_top, indicator]))

            z_bottom += z_change

        ground_truth_voxels.append(object_voxels)
        ground_truth_coords.append(object_coords)

    return np.array(ground_truth_voxels, dtype=object), \
        np.array(ground_truth_coords, dtype=object), \
        np.array(only_label_cloud_ind, dtype=object)


def add_noise(voxel):
    '''
    Add 3D Gaussian noise to a voxel
    '''
    return voxel


def rotate_voxel(voxel, coord):
    '''
    Rotate a voxel a random theta around the yaw axis
    '''
    return voxel, coord


# Number of swaps
Amin = 10
Amax = 200
# Number of voxels in each label to swap
Vmin = 1
Vmax = 6
# STD for adding noise
noise_std = 0.1


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
        coords (np.ndarray): (N, H, 7): bounds of the voxels in the first array:
            where N and H are the same as above
            and 7 represents mins [X, Y, Z], maxes [X, Y, Z], and label indicator
    
    Returns:
        np.ndarray: features parameter, after data augmentation
        np.ndarray: coords parameter, after data augmentation
    '''
    # for _ in range(randint(Amin, Amax)):
    for _ in range(1):
        # Allow label_1 = label_2, for intra-label augmentation
        label_1 = randint(0, features.shape[0] - 1)
        label_2 = randint(0, features.shape[0] - 1)
        print('label 1 idx', label_1, 'label 2 idx', label_2)

        label_1_deltas = np.array([abs(coords[label_1][0][3] - coords[label_1][0][0]),
                          abs(coords[label_1][0][4] - coords[label_1][0][1]),
                          abs(coords[label_1][0][5] - coords[label_1][0][2])])
        label_2_deltas = np.array([abs(coords[label_2][0][3] - coords[label_2][0][0]),
                          abs(coords[label_2][0][4] - coords[label_2][0][1]),
                          abs(coords[label_2][0][5] - coords[label_2][0][2])])

        num_swaps = randint(Vmin, Vmax)

        # Picks V random voxel indices in label_1 and label_2 (unordered)
        label_1_voxels = sample(range(0, len(features[label_1])), num_swaps)
        label_2_voxels = sample(range(0, len(features[label_2])), num_swaps)

        print('label 1 voxel idxs', label_1_voxels, 'label 2 voxel idxs', label_2_voxels)

        for label_1_voxel_idx, label_2_voxel_idx in zip(
                label_1_voxels, label_2_voxels):

            print('voxel 1', coords[label_1][label_1_voxel_idx])
            print('voxel 1 features\n', features[label_1][label_1_voxel_idx])

            # Calculate the centroids (average of max and min on each axis)
            voxel_1_centroid = np.mean(
                [coords[label_1][label_1_voxel_idx][3:-1], coords[label_1][label_1_voxel_idx][:3]], axis=0)
            voxel_2_centroid = np.mean(
                [coords[label_2][label_2_voxel_idx][3:-1], coords[label_2][label_2_voxel_idx][:3]], axis=0)

            # Move both voxels so they are centered around the origin
            features[label_1][label_1_voxel_idx] -= voxel_1_centroid
            features[label_2][label_2_voxel_idx] -= voxel_2_centroid

            # Resize the voxels
            features[label_1][label_1_voxel_idx] *= label_2_deltas / label_1_deltas
            features[label_2][label_2_voxel_idx] *= label_1_deltas / label_2_deltas

            # # Rotate the voxels
            # features[label_1][label_1_voxel_idx], coords[label_1][label_1_voxel_idx] = \
            #     rotate_voxel(features[label_1][label_1_voxel_idx], coords[label_1][label_1_voxel_idx])
            # features[label_2][label_2_voxel_idx], coords[label_2][label_2_voxel_idx] = \
            #     rotate_voxel(features[label_2][label_2_voxel_idx], coords[label_2][label_2_voxel_idx])

            # # Add noise
            # features[label_1][label_1_voxel_idx] = \
            #     add_noise(features[label_1][label_1_voxel_idx])
            # features[label_2][label_2_voxel_idx] = \
            #     add_noise(features[label_2][label_2_voxel_idx])

            # Apply the translation
            features[label_1][label_1_voxel_idx] += voxel_2_centroid
            features[label_2][label_2_voxel_idx] += voxel_1_centroid

            # Swap the voxels
            store_label_1_voxel = features[label_1][label_1_voxel_idx]
            features[label_1][label_1_voxel_idx] = features[label_2][label_2_voxel_idx]
            features[label_2][label_2_voxel_idx] = store_label_1_voxel

            # Swap the coordinates
            store_label_1_coords = coords[label_1][label_1_voxel_idx]
            coords[label_1][label_1_voxel_idx][:-1] = coords[label_2][label_2_voxel_idx][:-1]
            coords[label_2][label_2_voxel_idx][:-1] = store_label_1_coords[:-1]
    
    return features, coords


def display_voxels(coords, cloud, colors):
    all_voxel_boxes = []
    box_colors = []

    for i, label in enumerate(coords):
        for voxel in label:
            bounding_box = np.array(
                [[voxel[0], voxel[4], voxel[2]],  # min x, max y, min z
                 [voxel[0], voxel[1], voxel[2]],  # min x, min y, min z
                 [voxel[3], voxel[1], voxel[2]],  # max x, min y, min z
                 [voxel[3], voxel[4], voxel[2]],  # max x, max y, min z
                 [voxel[0], voxel[4], voxel[5]],  # min x, max y, max z
                 [voxel[0], voxel[1], voxel[5]],  # min x, min y, max z
                 [voxel[3], voxel[1], voxel[5]],  # max x, min y, max z
                 [voxel[3], voxel[4], voxel[5]]]  # max x, max y, max z
            )
            all_voxel_boxes.append(bounding_box)
            box_colors.append([colors[int(voxel[-1])] for _ in range(12)])

    visualize_lines_3d(pointcloud=cloud, gt_boxes=np.array(all_voxel_boxes),
                       gt_box_colors=box_colors, reduce_pts=True)


def features_to_cloud(features):
    cloud = np.empty((0, 3))
    for object in features:
        for voxel in object:
            cloud = np.concatenate((cloud, voxel), axis=0)
    return cloud


if __name__ == '__main__':
    cloud = o3d.io.read_point_cloud(sys.argv[1])
    points = np.asarray(cloud.points)
    lidar = filter_pointcloud(points, config='config_trunk')

    # Get an (X, 8, 3) array of the labels
    labels = load_custom_label(sys.argv[2])

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

    # # Visualize the entire point cloud with black bounding boxes
    # black = [0, 0, 0]
    # black_boxes = []
    # for _ in range(labels.shape[0]):
    #     black_boxes.append([black for _ in range(12)])

    # print('''Displaying the full point cloud and all labels:
    #     {} labels and {} points'''.format(labels.shape[0], lidar.shape[0]))
    # visualize_lines_3d(
    #     pointcloud=lidar, gt_boxes=labels, gt_box_colors=black_boxes, reduce_pts=False)

    # # Visualize only points inside the labels and the bounding boxes
    # color_boxes = []
    # for i in range(labels.shape[0]):
    #     color_boxes.append([label_colors[i] for _ in range(12)])

    # print('''Displaying only points inside labels:
    #     {} labels and {} points'''.format(labels.shape[0], only_label_pts.shape[0]))
    # visualize_lines_3d(
    #     pointcloud=only_label_pts, gt_boxes=labels, gt_box_colors=color_boxes,
    #     reduce_pts=True)

    # print('Displaying the voxelized labels')
    # display_voxels(
    #     coords, cloud=only_label_pts, colors=label_colors)

    augmented_features, augmented_coords = aug_data(features, coords)

    augmented_cloud = features_to_cloud(augmented_features)

    print('Displaying augmented data')
    display_voxels(
        coords, cloud=augmented_cloud, colors=label_colors)