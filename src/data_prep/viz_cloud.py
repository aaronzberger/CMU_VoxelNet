'''
Visualize a point cloud
'''


import numpy as np
import sys
import os
from random import uniform

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz_3d import visualize_lines_3d
from conversions import load_numpy_label
from utils import snap_labels
from config import data_dir
from data_prep.data_aug import voxelize


if __name__ == '__main__':

    data_path = data_dir + '/cropped/' + sys.argv[1]

    data = np.load(data_path)

    pointcloud, labels = data['pointcloud'], data['labels']

    labels = load_numpy_label(labels)
    try:
        corner_boxes = snap_labels(pointcloud, labels)
    except ValueError:
        print('At least one label contained no points:' +
              'the labels were probably saved incorrectly')
        sys.exit(2)

    print('''
    Min X: {}   Max X: {}
    Min Y: {}   Max Y: {}
    Min Z: {}   Max Z: {}
    '''.format(min(pointcloud[:, 0]), max(pointcloud[:, 0]),
               min(pointcloud[:, 1]), max(pointcloud[:, 1]),
               min(pointcloud[:, 2]), max(pointcloud[:, 2])))

    # Voxelize the labels
    features, coords, cloud_ind = voxelize(pointcloud, labels)

    # Separate the points by whether they are in a label
    cloud_ind = cloud_ind.astype(int)
    only_label_pts = pointcloud[cloud_ind]
    non_label_pts = np.delete(pointcloud, cloud_ind, axis=0)

    # Generate a list of colors, one for each object instance (N, 3),
    # where N is the number of labels, and 3 encodes [R, G, B]
    label_colors = []
    for _ in range(corner_boxes.shape[0]):
        label_colors.append(
            [uniform(0.0, 0.9), uniform(0.0, 0.9), uniform(0.0, 0.9)])

    # Visualize the entire point cloud with black bounding boxes
    black = [0, 0, 0]
    black_boxes = []
    for _ in range(corner_boxes.shape[0]):
        black_boxes.append([black for _ in range(12)])

    color_boxes = []
    for i in range(corner_boxes.shape[0]):
        color_boxes.append([label_colors[i] for _ in range(12)])

    print('''Displaying the full point cloud and all labels:
        {} labels and {} points'''.format(
            labels.shape[0], pointcloud.shape[0]))
    visualize_lines_3d(
        pointcloud=pointcloud, gt_boxes=corner_boxes,
        gt_box_colors=black_boxes, reduce_pts=False)

    print('''Displaying only label points and all labels:
        {} labels and {} points'''.format(
            labels.shape[0], only_label_pts.shape[0]))
    visualize_lines_3d(
        pointcloud=only_label_pts, gt_boxes=corner_boxes,
        gt_box_colors=color_boxes, reduce_pts=False)
