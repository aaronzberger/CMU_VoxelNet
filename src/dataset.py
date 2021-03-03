from __future__ import division
import os
import os.path
import torch.utils.data as data

from box_overlaps import bbox_overlaps
import numpy as np
import cv2

from utils import load_config, get_num_voxels, get_anchors
from utils import load_kitti_label, load_kitti_calib, filter_pointcloud
from config import data_dir
from conversions import box3d_corner_to_center_batch, anchors_center_to_corner
from conversions import corner_to_standup_box2d


class KittiDataset(data.Dataset):
    def __init__(self, split='train'):
        self.config = load_config()

        self.data_path = os.path.join(data_dir, '%sing' % split)
        self.lidar_path = os.path.join(self.data_path, "cropped/")
        self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")

        # Open the split file, containing numbers of the examples in this split
        # with open(os.path.join(self.data_path, '%s.txt' % split)) as f:
        with open(os.path.join(self.data_path, 'test_net.txt')) as f:
            self.file_list = f.read().splitlines()

        # Calculate the size of the voxel grid in every dimensions
        self.voxel_W, self.voxel_H, self.voxel_D = get_num_voxels()

        self.anchors = get_anchors().reshape(-1, 7)

        self.feature_map_shape = (self.voxel_H // 2, self.voxel_W // 2)

    def cal_target(self, gt_box3d):
        '''
        Calculate the positive and negative anchors, and the target

        Parameters:
            gt_box3d (arr): (N, 8, 3)
                ground truth bounding boxes in corners notation

        Returns:
            arr: positive anchor positions
            arr: negative anchor positions
            arr: targets
        '''
        #       _______________
        # dᵃ = √ (lᵃ)² + (wᵃ)²      is the diagonal of the base
        #                           of the anchor box (See 2.2)
        anchors_diagonal = np.sqrt(
            self.anchors[:, 4] ** 2 + self.anchors[:, 5] ** 2)

        pos_equal_one = np.zeros((*self.feature_map_shape, 2))
        neg_equal_one = np.zeros((*self.feature_map_shape, 2))

        # Convert from corner to center notation ((N, 8, 3) -> (N, 7))
        gt_xyzhwlr = box3d_corner_to_center_batch(gt_box3d)

        # Convert anchors to corner notation (BEV)
        anchors_corner = anchors_center_to_corner(self.anchors)

        # Convert to from all corners to only 2 [xyxy]
        anchors_standup_2d = corner_to_standup_box2d(anchors_corner)
        gt_standup_2d = corner_to_standup_box2d(gt_box3d)

        # Calculate IoU of the ground truth and anchors (BEV)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        # Indices of X highest anchors by IoU, X = number of ground truths
        id_highest = np.argmax(iou.T, axis=1)

        # Array containg [0, 1, 2, ..., X-1], X = number of ground truths
        id_highest_gt = np.arange(iou.T.shape[0])

        # Make sure the anchor we picked has an IoU > 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # An anchor is considered as positive if it has the highest IoU
        # with a ground truth or its IoU with ground truth is ≥ pos_threshold
        # (in BEV). (See 3.1)

        # id_pos: Index of anchor        id_pos_gt: Index of ground truth
        id_pos, id_pos_gt = np.where(iou > self.config['IOU_pos_threshold'])

        # An anchor is considered as negative if the IoU between it and all
        # ground truth boxes is less than neg_threshold. (See 3.1)
        id_neg = np.where(np.sum(iou < self.config['IOU_neg_threshold'],
                                 axis=1) == iou.shape[1])[0]
        id_neg.sort()

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        # Filter out repeats (above pos_threshold and max)
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]

        # Calculate target (𝘂*) and set corresponding feature map spaces to 1
        index_x, index_y, index_z = np.unravel_index(
            id_pos,
            (*self.feature_map_shape, self.config['anchors_per_position']))
        pos_equal_one[index_x, index_y, index_z] = 1

        # To retrieve the ground truth box from a matching positive anchor,
        # we define the residual vector 𝘂* ("targets") containing the 7
        # regression targets corresponding to center location ∆x,∆y,∆z,
        # three dimensions ∆l,∆w,∆h, and the rotation ∆θ (See 2.2)
        targets = np.zeros((*self.feature_map_shape, 14))

        # Δx = (xᵍ - xᵃ) / dᵃ
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_xyzhwlr[id_pos_gt, 0] - self.anchors[id_pos, 0]) \
            / anchors_diagonal[id_pos]

        # Δy = (yᵍ - yᵃ) / dᵃ
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_xyzhwlr[id_pos_gt, 1] - self.anchors[id_pos, 1]) \
            / anchors_diagonal[id_pos]

        # Δz = (zᵍ - zᵃ) / hᵃ
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_xyzhwlr[id_pos_gt, 2] - self.anchors[id_pos, 2]) \
            / self.anchors[id_pos, 3]

        # Δh = log(hᵍ / hᵃ)
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = \
            np.log(gt_xyzhwlr[id_pos_gt, 3] / self.anchors[id_pos, 3])

        # Δw = log(wᵍ / wᵃ)
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = \
            np.log(gt_xyzhwlr[id_pos_gt, 4] / self.anchors[id_pos, 4])

        # Δl = log(lᵍ / lᵃ)
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = \
            np.log(gt_xyzhwlr[id_pos_gt, 5] / self.anchors[id_pos, 5])

        # Δ𝜃 = 𝜃ᵍ - 𝜃ᵃ
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
            gt_xyzhwlr[id_pos_gt, 6] - self.anchors[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(
            id_neg,
            (*self.feature_map_shape, self.config['anchors_per_position']))
        neg_equal_one[index_x, index_y, index_z] = 1

        # To avoid a box being positive and negative
        index_x, index_y, index_z = np.unravel_index(
            id_highest,
            (*self.feature_map_shape, self.config['anchors_per_position']))
        neg_equal_one[index_x, index_y, index_z] = 0

        return pos_equal_one, neg_equal_one, targets

    def voxelize(self, lidar):
        '''
        Convert an input point cloud into a voxelized point cloud

        Parameters:
            lidar (arr): point cloud

        Returns:
            arr: list of all the voxels, each arrays containing points
            arr: coordinates of voxels in the first return array
        '''
        # Shuffle the points
        np.random.shuffle(lidar)

        voxel_coords = ((lidar[:, :3] - np.array(
            [self.config['pcl_range']['X1'], self.config['pcl_range']['Y1'],
             self.config['pcl_range']['Z1']])) / (
            self.config['voxel_size']['W'], self.config['voxel_size']['H'],
            self.config['voxel_size']['D'])).astype(np.int32)

        # Convert to (D, H, W)
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        # Get info on the non-empty voxels
        voxel_coords, inv_ind, voxel_counts = np.unique(
            voxel_coords, axis=0, return_inverse=True, return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros(
                (self.config['max_pts_per_voxel'], 7), dtype=np.float32)

            # inv_ind gives the indices of the elements in the original array
            pts = lidar[inv_ind == i]

            if voxel_counts[i] > self.config['max_pts_per_voxel']:
                pts = pts[:self.config['max_pts_per_voxel'], :]
                voxel_counts[i] = self.config['max_pts_per_voxel']

            # Augment each point with its relative offset
            # w.r.t. the centroid of this voxel (See 2.1.1)
            voxel[:pts.shape[0], :] = np.concatenate(
                (pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)

        return np.array(voxel_features), voxel_coords

    def __getitem__(self, i):
        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        image_file = self.image_path + '/' + self.file_list[i] + '.png'

        # Load the calibration file and the transform from pcl to image
        calib = load_kitti_calib(calib_file)
        Tr = calib['Tr_velo2cam']

        # Load the GT boxes
        gt_box3d = load_kitti_label(label_file, Tr)

        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        image = cv2.imread(image_file)

        # Crop the lidar
        lidar, gt_box3d = filter_pointcloud(lidar, gt_box3d)

        # visualize_lines_3d(gt_box3d, lidar)

        # Voxelize
        voxel_features, voxel_coords = self.voxelize(lidar)

        # Calculate positive and negative anchors, and GT target
        pos_equal_one, neg_equal_one, targets = self.cal_target(gt_box3d)

        return voxel_features, voxel_coords, pos_equal_one, \
            neg_equal_one, targets, gt_box3d, lidar, image, calib, \
            self.file_list[i]

    def __len__(self):
        return len(self.file_list)


def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    pos_equal_one = []
    neg_equal_one = []
    targets = []
    gt_bounding_boxes = []
    lidars = []

    images = []
    calibs = []
    ids = []

    for i, sample in enumerate(batch):
        voxel_features.append(sample[0])

        # Pad which number in the batch this is
        # to the beginning of each voxel array (Axis 1)
        voxel_coords.append(
            np.pad(sample[1], ((0, 0), (1, 0)),
                   mode='constant', constant_values=i))

        pos_equal_one.append(sample[2])
        neg_equal_one.append(sample[3])
        targets.append(sample[4])
        gt_bounding_boxes.append(sample[5])
        lidars.append(sample[6])

        images.append(sample[7])
        calibs.append(sample[8])
        ids.append(sample[9])

    return np.concatenate(voxel_features), np.concatenate(voxel_coords), \
        np.array(pos_equal_one), np.array(neg_equal_one), np.array(targets), \
        np.array(gt_bounding_boxes), np.array(lidars), images, calibs, ids


def get_data_loader():
    train_dataset = KittiDataset(split='train')
    train_data_loader = data.DataLoader(
        train_dataset, shuffle=True, batch_size=load_config()['batch_size'],
        num_workers=8, collate_fn=detection_collate)

    return train_data_loader, len(train_data_loader)
