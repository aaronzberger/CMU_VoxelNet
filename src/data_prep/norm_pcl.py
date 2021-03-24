'''
Normalize the pointclouds and labels (center around the origin and crop)
in preparation for VoxelNet training

Usage: norm_pcl.py
use "norm_pcl.py viz" if you wish to view the updated pointclouds
    and labels as well writing them
'''


import open3d as o3d
import numpy as np
from sys import float_info
import sys
from viz_3d import visualize_lines_3d
from conversions import box3d_center_to_corner
from utils import snap_labels
from config import data_dir


if __name__ == '__main__':
    min_f, max_f = -float_info.max, float_info.max
    crops = np.array([
        #      X               Y               Z
        #  min    max      min    max      min    max
        [[min_f, max_f], [min_f, max_f], [min_f, max_f]],
        [[min_f, max_f], [min_f, max_f], [min_f, max_f]],
        [[-28, max_f], [min_f, max_f], [min_f, max_f]],
        [[min_f, max_f], [min_f, max_f], [min_f, max_f]],
        [[min_f, max_f], [min_f, max_f], [min_f, max_f]],
        [[min_f, max_f], [min_f, max_f], [min_f, max_f]],
        [[min_f, max_f], [-70, max_f], [min_f, max_f]],
        [[min_f, max_f], [-80, max_f], [min_f, max_f]],
    ])

    for i in range(1, 9):
        cloud_path = data_dir + '/cloud_{}/cloud_{}.pcd'.format(i, i)
        label_path = data_dir + '/cloud_{}/cloud_{}_labels.txt'.format(i, i)

        save_path = data_dir + '/cropped/cloud_{}'.format(i)

        cloud = o3d.io.read_point_cloud(cloud_path)
        points = np.asarray(cloud.points)

        # Indices of in-bound points for every dimension
        valid_x = np.where((points[:, 0] > crops[i - 1][0][0])
                           & (points[:, 0] < crops[i - 1][0][1]))[0]
        valid_y = np.where((points[:, 1] > crops[i - 1][1][0])
                           & (points[:, 1] < crops[i - 1][1][1]))[0]
        valid_z = np.where((points[:, 2] > crops[i - 1][2][0])
                           & (points[:, 2] < crops[i - 1][2][1]))[0]

        # Combine the index arrays and filter
        valid_xyz = np.intersect1d(valid_z, np.intersect1d(valid_x, valid_y))
        points = points[valid_xyz]

        # Center the pointcloud around the origin
        min_x, max_x = min(points[:, 0]), max(points[:, 0])
        min_y, max_y = min(points[:, 1]), max(points[:, 1])
        min_z, max_z = min(points[:, 2]), max(points[:, 2])

        shift_x = (max_x + min_x) / 2
        shift_y = (max_y + min_y) / 2
        shift_z = min_z

        points[:, 0] -= shift_x
        points[:, 1] -= shift_y
        points[:, 2] -= shift_z

        print('''
        Min X: {}   Max X: {}
        Min Y: {}   Max Y: {}
        Min Z: {}   Max Z: {}
        '''.format(min(points[:, 0]), max(points[:, 0]),
                   min(points[:, 1]), max(points[:, 1]),
                   min(points[:, 2]), max(points[:, 2])))

        # Shift the labels around the origin
        updated_labels = []

        with open(label_path, 'r') as label_file:
            lines = label_file.readlines()

        for line in lines:
            line = line.split(' ')
            name = line[0]
            tx, ty, tz, l, w, h, ry = [float(i) for i in line[1:]]

            tx -= shift_x
            ty -= shift_y
            tz -= shift_z

            updated_labels.append([name, tx, ty, tz, l, w, h, ry])

        if sys.argv[1] == 'viz':
            corner_boxes = []

            for label in updated_labels:
                tx, ty, tz, l, w, h, ry = [float(i) for i in label[1:]]

                box = np.expand_dims(
                    np.array([tx, ty, tz, h, w, l, ry]), axis=0)

                # Transform label into corner notation bounding box
                box3d_corner = box3d_center_to_corner(box, z_middle=True)

                corner_boxes.append(box3d_corner)

            corner_boxes = np.array(corner_boxes).reshape(-1, 8, 3)

            corner_boxes = snap_labels(points, corner_boxes)

            visualize_lines_3d(pointcloud=points, gt_boxes=corner_boxes)

        np.savez(save_path, pointcloud=points, labels=updated_labels)
        print('Saved cropped pointcloud and labels to {}'.format(save_path))
