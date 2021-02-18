import os

import open3d as o3d
import numpy as np
import sys

from utils import mkdir_p
from config import data_dir


def viz_all():
    files = os.listdir(os.path.join(data_dir, 'viz'))
    files = sorted(list(files), key=lambda x: str(x[:-4]))
    for f in files:
        viz_file = np.load(os.path.join(data_dir, 'viz', f))
        print('Visualizing PointCloud {} from Epoch {}'.format(
            viz_file['desc'][0], viz_file['desc'][1]))
        visualize_lines_3d(boxes=viz_file['boxes'],
                           pointcloud=viz_file['pointcloud'],
                           gt_boxes=viz_file['gt_boxes'])


def save_center_batch(batch_boxes, batch_lidar, epoch, ids, gt_boxes=None):
    for i in range(batch_boxes.shape[0]):
        print('output had {} boxes'.format(batch_boxes[i].shape[0]))
        boxes = batch_boxes[i]
        save_viz_file(
            boxes=boxes, pointcloud=batch_lidar[i],
            name="epoch" + str(epoch) + "_pcl" + str(ids[i]),
            desc=[ids[i], epoch], gt_boxes=gt_boxes)


def save_viz_file(boxes, pointcloud, name, desc, gt_boxes):
    '''
    Save a numpy file containing the boxes and pointcloud

    Parameters:
        boxes (arr): boxes (currently only corner notation supported)
        pointcloud (arr): raw pointcloud: N,3 or N,4 array
        name (str): name of the file
    '''
    if len(pointcloud.shape) != 2 or pointcloud.shape[1] < 3:
        raise ValueError(
            'pointcloud argument must be (N,3) or (N,4) array, was of shape', pointcloud.shape)
    if boxes.shape[1] != 8 or boxes.shape[2] != 3:
        raise ValueError(
            'boxes argument must be (N,8,3) array, was of shape', boxes.shape)
    mkdir_p(os.path.join(data_dir, 'viz'))
    np.savez(os.path.join(data_dir, 'viz', name),
             boxes=boxes, pointcloud=pointcloud, desc=desc, gt_boxes=gt_boxes)


def visualize_lines_3d(boxes, pointcloud, gt_boxes=None):
    '''
    Use Open3D to plot labels on 3D point clouds

    Parameters:
        boxes (arr): boxes in corner notation
        pointcloud (arr): raw pointcloud
    '''
    gt_boxes = gt_boxes[0]
    if len(pointcloud.shape) != 2 or pointcloud.shape[1] < 3:
        raise ValueError('pointcloud argument must be (N,3) or (N,4) array')
    if boxes.shape[1] != 8 or boxes.shape[2] != 3:
        raise ValueError('boxes argument must be (N,8,3) array')

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_sets = []
    for box in boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)

    colors = [[0, 0, 1] for _ in range(len(lines))]

    for box in gt_boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)

    print('PREDICTIONS')
    print(boxes)
    print('GROUND TRUTH')
    print(gt_boxes)

    pcl = o3d.geometry.PointCloud()

    pcl.points = o3d.utility.Vector3dVector(
        pointcloud[:, :3].astype('float64'))

    o3d.visualization.draw_geometries([
        pcl, *line_sets
    ])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for item in sys.argv[1:]:
            viz_file = np.load(item)
            print('Visualizing PointCloud {} from Epoch {}, containing {} ground truth boxes'.format(
                viz_file['desc'][0], viz_file['desc'][1], len(viz_file['gt_boxes'][0])))
            visualize_lines_3d(boxes=viz_file['boxes'],
                               pointcloud=viz_file['pointcloud'],
                               gt_boxes=viz_file['gt_boxes'])
    else:
        viz_all()
