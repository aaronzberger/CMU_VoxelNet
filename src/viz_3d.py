import os

import open3d as o3d
import numpy as np
import sys

from utils import mkdir_p
from conversions import load_custom_label
from config import base_dir


def viz_all(random=False):
    '''
    Visualize all results in the viz folder

    Parameters:
        random (bool): whether to show them in a random order
    '''
    files = os.listdir(os.path.join(base_dir, 'viz'))
    if not random:
        files = sorted(list(files), key=lambda x: str(x[:-4]))
    for f in files:
        viz_file = np.load(os.path.join(base_dir, 'viz', f))
        visualize_file(viz_file)


def visualize_file(np_file):
    '''
    Run visualization code for a loaded numpy file and print info

    Parameters:
        np_file: loaded .npz file in dict format
    '''
    print('Visualizing point cloud  {}  from epoch  {} '.format(
        np_file['desc'][0], np_file['desc'][1]) +

           ' containing  {}  ground truth boxes'.format(
               np_file['gt_boxes'].shape[1]) +

           ' and  {}  predictions'.format(
               np_file['boxes'].shape[0]))

    visualize_lines_3d(boxes=np_file['boxes'],
                       pointcloud=np_file['pointcloud'],
                       gt_boxes=np_file['gt_boxes'])


def save_viz_batch(batch_boxes, batch_lidar, epoch, ids, gt_boxes=None):
    '''
    Save a batch of visualizations

    Parameters:
        batch_boxes (arr): the prediction boxes
        batch_lidar (arr): the point cloud
        epoch (str): description of the batch (usually epoch number)
        ids (arr): list of ids for the viz file
        gt_boxes (arr): ground truth bounding boxes, if desired
    '''
    for i in range(batch_lidar.shape[0]):
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
            'pointcloud argument must be (N,3) or (N,4) array, was of shape',
            pointcloud.shape)

    if boxes.shape[1] != 8 or boxes.shape[2] != 3:
        raise ValueError(
            'boxes argument must be (N,8,3) array, was of shape',
            boxes.shape)

    # Ensure the dir exists, then save the file
    mkdir_p(os.path.join(base_dir, 'viz'))
    np.savez(os.path.join(base_dir, 'viz', name),
             boxes=boxes, pointcloud=pointcloud, desc=desc, gt_boxes=gt_boxes)


def visualize_lines_3d(pointcloud, boxes=None, gt_boxes=None):
    '''
    Use Open3D to plot labels on 3D point clouds

    Parameters:
        boxes (arr): boxes in corner notation
        pointcloud (arr): raw pointcloud
    '''
    if isinstance(pointcloud, np.ndarray):
        if len(pointcloud.shape) != 2 or pointcloud.shape[1] < 3:
            raise ValueError(
                'pointcloud must be of shape (N,3) or (N,4), was of shape',
                pointcloud.shape)

    if boxes is not None:
        if boxes.shape[1] != 8 or boxes.shape[2] != 3:
            raise ValueError(
                'boxes argument must be (N,8,3) array, was of shape',
                boxes.shape)

    if len(gt_boxes.shape) == 4:
        gt_boxes = gt_boxes[0]
    if gt_boxes is not None:
        if gt_boxes.shape[1] != 8 or gt_boxes.shape[2] != 3:
            raise ValueError(
                'gt_boxes argument must be (N,8,3) array, was of shape',
                gt_boxes.shape)

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_sets = []
    if boxes is not None:
        for box in boxes:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)

    colors = [[0, 0, 0] for _ in range(len(lines))]

    if gt_boxes is not None:
        for box in gt_boxes:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().load_from_json("/home/aaron/options.json")

    # print(gt_boxes)
    print(len(gt_boxes))

    if isinstance(pointcloud, np.ndarray):
        pcl = o3d.geometry.PointCloud()

        pcl.points = o3d.utility.Vector3dVector(
            pointcloud[:, :3].astype('float64'))
    else:
        pcl = pointcloud

    vis.add_geometry(pcl)
    for tree in line_sets:
        vis.add_geometry(tree)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    if sys.argv[1] not in ['all', 'these', 'custom']:
        raise ValueError('''
    First argument must be either \'all\', \'these\', or \'custom\'.

    all: visualize all point clouds and boxes in the folder {}
    these: the next arguments should be individual paths to viz files
        (probably in the folder {})
    custom: the next arguments should be a point cloud file and labels file
    '''.format(os.path.join(base_dir, 'viz'), os.path.join(base_dir, 'viz'), ))

    if sys.argv[1] == 'custom':
        cloud = o3d.io.read_point_cloud(sys.argv[1])
        labels = load_custom_label(sys.argv[2])
        visualize_lines_3d(pointcloud=cloud, gt_boxes=labels)
    elif sys.argv[1] == 'these':
        for item in sys.argv[1:]:
            viz_file = np.load(item)
            visualize_file(viz_file)
    else:
        viz_all()
