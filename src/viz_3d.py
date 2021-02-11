import os

import open3d as o3d
import numpy as np

from utils import mkdir_p
from config import data_dir


def viz_all():
    files = os.listdir(os.path.join(data_dir, 'viz'))
    files = sorted(list(files), key=lambda x: int(x[:-4]))
    for f in files:
        print(f)


def save_viz_file(boxes, pointcloud, id, desc):
    '''
    Save a numpy file containing the boxes and pointcloud

    Parameters:
        boxes (arr): boxes (currently only corner notation supported)
        pointcloud (arr): raw pointcloud: N,3 or N,4 array
        id (int): id to save the file
    '''
    if len(pointcloud.shape) != 2 or pointcloud.shape[1] < 3:
        raise ValueError('pointcloud argument must be (N,3) or (N,4) array')
    if boxes.shape[1] != 8 or boxes.shape[2] != 3:
        raise ValueError('boxes argument must be (N,8,3) array')

    mkdir_p(os.path.join(data_dir, 'viz'))
    np.savez(os.path.join(data_dir, 'viz', id),
             boxes=boxes, pointcloud=pointcloud, desc=desc)


def viz_from_file(id):
    '''
    Visualize the boxes in the pointcloud from a saved file

    Parameters:
        id (int): the id of the image (to load the file)
    '''
    viz_file = np.load(os.path.join(data_dir, 'viz', id))
    print('Visualizing ID {}: {}'.format(id, viz_file['desc']))
    visualize_lines_3d(boxes=viz_file['gt_boxes'],
                       pointcloud=viz_file['pointcloud'])


def visualize_lines_3d(boxes, pointcloud):
    '''
    Use Open3D to plot labels on 3D point clouds

    Parameters:
        boxes (arr): boxes in corner notation
        pointcloud (arr): raw pointcloud
    '''

    if len(pointcloud.shape) != 2 or pointcloud.shape[1] < 3:
        raise ValueError('pointcloud argument must be (N,3) or (N,4) array')
    if boxes.shape[1] != 8 or boxes.shape[2] != 3:
        raise ValueError('boxes argument must be (N,8,3) array')

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0] for i in range(len(lines))]

    line_sets = []
    for box in boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)

    pcl = o3d.geometry.PointCloud()

    pcl.points = o3d.utility.Vector3dVector(
        pointcloud[:, :3].astype('float64'))

    o3d.visualization.draw_geometries([
        pcl, *line_sets
    ])
