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
               np_file['gt_boxes'].shape[0]) +

           ' and  {}  predictions'.format(
               np_file['boxes'].shape[0]))

    visualize_lines_3d(pointcloud=np_file['pointcloud'],
                       boxes=np_file['boxes'],
                       gt_boxes=np_file['gt_boxes'])


def save_viz_batch(pointcloud, boxes, gt_boxes, epoch, ids):
    '''
    Save a batch of visualizations

    Parameters:
        pointcloud (arr): (N, X, 3) the point cloud
            where N is the batch size, X is the number of points
        boxes (arr): (N, Y, 8, 3) the prediction boxes
            where Y is the number of boxes and (8, 3) are the corner points
        gt_boxes (arr): (N, J, 8, 3) ground truth bounding boxes
        epoch (str): description of the batch (usually epoch number)
        ids (arr): list of ids for the viz file name
    '''
    for i in range(pointcloud.shape[0]):
        save_viz_file(
            pointcloud=pointcloud[i],
            boxes=boxes[i], gt_boxes=gt_boxes[i],
            name="epoch" + str(epoch) + "_pcl" + str(ids[i]),
            desc=[ids[i], epoch])


def save_viz_file(pointcloud, boxes, gt_boxes, name, desc):
    '''
    Save a numpy file containing the boxes and pointcloud

    Parameters:
        pointcloud (arr): raw pointcloud: (N, 3) or (N, 4)
        boxes (arr): (Y, 8, 3) boxes
        gt_boxes (arr): (J, 8, 3) ground truth boxes
        name (str): name of the file
        desc (str): description to display when visualizing this file
    '''
    if len(pointcloud.shape) != 2 or pointcloud.shape[1] < 3:
        raise ValueError(
            'pointcloud argument must be (N, 3) or (N, 4) array, was of shape',
            pointcloud.shape)

    if boxes.shape[1] != 8 or boxes.shape[2] != 3:
        raise ValueError(
            'boxes argument must be (N, 8, 3) array, was of shape',
            boxes.shape)

    # Ensure the folder exists, then save the file into it
    mkdir_p(os.path.join(base_dir, 'viz'))
    np.savez(os.path.join(base_dir, 'viz', name),
             pointcloud=pointcloud, boxes=boxes, gt_boxes=gt_boxes, desc=desc)


def visualize_lines_3d(
        pointcloud=None, boxes=None, gt_boxes=None,
        box_colors=None, gt_box_colors=None, reduce_pts=False):
    '''
    Use Open3D to plot labels on 3D point clouds

    Parameters:
        pointcloud (arr): (N, 3) or (N, 4) raw pointcloud
        boxes (arr): (Y, 8, 3) boxes in corner notation
        gt_boxes (arr): (J, 8, 3) boxes in corner notation
        box_colors (arr): (K, 12) colors of the lines in boxes param
        gt_box_colors (arr): (L, 12) colors of the lines in gt_boxes param
        reduce_pts (bool): whether to shrink the points and display them grey
            (to help visualize other objects)
    '''
    if pointcloud is not None:
        if isinstance(pointcloud, list):
            pointcloud = np.array(pointcloud)
        if len(pointcloud.shape) != 2 or pointcloud.shape[1] < 3:
            raise ValueError(
                'pointcloud must be of shape (N, 3) or (N, 4), was of shape',
                pointcloud.shape)

    if boxes is not None:
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        if len(boxes.shape) == 4:
            boxes = boxes[0]
        if boxes.shape[1] != 8 or boxes.shape[2] != 3:
            raise ValueError(
                'boxes argument must be (N, 8, 3) array, was of shape',
                boxes.shape)

    if gt_boxes is not None:
        if isinstance(gt_boxes, list):
            gt_boxes = np.array(gt_boxes)
        if len(gt_boxes.shape) == 4:
            gt_boxes = gt_boxes[0]
        if gt_boxes.shape[1] != 8 or gt_boxes.shape[2] != 3:
            raise ValueError(
                'gt_boxes argument must be (N, 8, 3) array, was of shape',
                gt_boxes.shape)

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # If box_colors wasn't specified, display predictions red
    if box_colors is None and boxes is not None:
        box_colors = [[[0.8, 0, 0] for _ in range(len(lines))]
                      for _ in range(boxes.shape[0])]

    line_sets = []
    if boxes is not None:
        for i, box in enumerate(boxes):
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(box_colors[i])
            line_sets.append(line_set)

    # If gt_box_colors wasn't specified, display ground truths green
    if gt_box_colors is None and gt_boxes is not None:
        gt_box_colors = [[[0, 0.8, 0] for _ in range(len(lines))]
                         for _ in range(gt_boxes.shape[0])]

    if gt_boxes is not None:
        for i, box in enumerate(gt_boxes):
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(box)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(gt_box_colors[i])
            line_sets.append(line_set)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if pointcloud is not None:
        if isinstance(pointcloud, np.ndarray):
            pcl = o3d.geometry.PointCloud()

            pcl.points = o3d.utility.Vector3dVector(
                pointcloud[:, :3].astype('float64'))
        else:
            pcl = pointcloud

        # Color the points grey if we want to reduce their importance
        if reduce_pts:
            pcl = pcl.paint_uniform_color(np.array([0.5, 0.5, 0.5]))

        vis.add_geometry(pcl)

    for line in line_sets:
        vis.add_geometry(line)

    options = vis.get_render_option()

    # Decrease point size if we want to reduce point importance
    options.point_size = 2.0 if reduce_pts else 5.0

    vis.update_renderer()

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
        cloud = o3d.io.read_point_cloud(sys.argv[2])
        print(cloud.get_min_bound(), cloud.get_max_bound())
        points = np.asarray(cloud.points)
        print(points.shape)
        labels = load_custom_label(sys.argv[3])
        visualize_lines_3d(pointcloud=cloud, gt_boxes=labels)
    elif sys.argv[1] == 'these':
        for item in sys.argv[2:]:
            viz_file = np.load(item)
            visualize_file(viz_file)
    else:
        viz_all()
