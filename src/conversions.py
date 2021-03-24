import torch
import numpy as np

from nms import nms
from utils import get_anchors, load_config


def box3d_center_to_corner(boxes_center, z_middle=False):
    '''
    Transform bounding boxes from center to corner notation

    Parameters:
        boxes_center (arr): (X, 7):
            boxes in center notation [xyzhwlr]
        z_middle (bool): whether the z in boxes_center is at the middle of
            the object (it's normally at the bottom for KITTI labels)

    Returns:
        arr: bounding box in corner notation
    '''
    if torch.is_tensor(boxes_center):
        if boxes_center.is_cuda:
            boxes_center = boxes_center.cpu().numpy()
    num_boxes = boxes_center.shape[0]

    # To return
    corner_boxes = np.zeros((num_boxes, 8, 3))

    for box_num, box in enumerate(boxes_center):
        translation = box[0:3]
        h, w, l = box[3], box[4], box[5]
        rotation = box[6]

        # Create the bounding box outline (to be transposed)
        bounding_box = np.array([
            [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
            [0, 0, 0, 0, h, h, h, h]])
        if z_middle:
            bounding_box[2] = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

        # Standard 3x3 rotation matrix around the Z axis
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0.0],
            [np.sin(rotation), np.cos(rotation), 0.0],
            [0.0, 0.0, 1.0]])

        # Repeat [x, y, z] eight times
        eight_points = np.tile(translation, (8, 1))

        # Translate the rotated bounding box by the
        # original center position to obtain the final box
        corner_box = np.dot(
            rotation_matrix, bounding_box) + eight_points.transpose()
        corner_box = corner_box.transpose()

        corner_boxes[box_num] = corner_box

    return corner_boxes


def box3d_center_to_corner_batch(boxes_center):
    '''
    Transform bounding boxes from center to corner notation
    by calling the box3d_center_to_corner method over the batch

    Parameters:
        boxes_center (arr): (N, X, 7):
            boxes in center notation

    Returns:
        arr: bounding box in corner notation
    '''
    if torch.is_tensor(boxes_center):
        if boxes_center.is_cuda:
            boxes_center = boxes_center.cpu().numpy()
    batch_size = boxes_center.shape[0]
    num_boxes = boxes_center.shape[1]

    # To return
    corner_boxes = np.zeros((batch_size, num_boxes, 8, 3))

    for batch_id in range(batch_size):
        corner_boxes[batch_id] = box3d_center_to_corner(boxes_center[batch_id])

    return corner_boxes


def load_custom_label(label_file):
    '''
    Load the custom label file for an example

    Parameters:
        label_file (str): label file full path

    Returns:
        arr: array containing GT boxes in the corner notation
    '''
    config = load_config()

    with open(label_file, 'r') as f:
        lines = f.readlines()

    gt_boxes3d_corner = []

    for j in range(len(lines)):
        obj = lines[j].strip().split(' ')

        # Ensure the GT class is one we're using
        if obj[0].strip() not in config['class_list']:
            continue

        # w, h, l, tx, ty, tz, ry = [float(i) for i in obj[1:]]
        tx, ty, tz, l, w, h, ry = [float(i) for i in obj[1:]]

        box = np.expand_dims(np.array([tx, ty, tz, h, w, l, ry]), axis=0)

        # Transform label into coordinates of 8 points that make up the bbox
        box3d_corner = box3d_center_to_corner(box, z_middle=True)

        gt_boxes3d_corner.append(box3d_corner)

    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1, 8, 3)

    return gt_boxes3d_corner


def load_numpy_label(labels):
    corner_boxes = []

    for label in labels:
        tx, ty, tz, l, w, h, ry = [float(i) for i in label[1:]]

        box = np.expand_dims(np.array([tx, ty, tz, h, w, l, ry]), axis=0)

        # Transform label into coordinates of 8 points that make up the bbox
        box3d_corner = box3d_center_to_corner(box, z_middle=True)

        corner_boxes.append(box3d_corner)

    corner_boxes = np.array(corner_boxes).reshape(-1, 8, 3)

    return corner_boxes


def anchors_center_to_corner(anchors):
    '''
    Convert anchors to corner notation (BEV only)
    (N, 7) -> (N, 4, 3)

    Parameters:
        anchors (arr): the anchors in the form [xyzhwlr]

    Returns:
        arr: the anchors in corner notation
    '''
    N = anchors.shape[0]

    anchors_corner = np.zeros((N, 4, 2))

    for i in range(N):
        anchor = anchors[i]
        h, w, l = anchor[3:6]
        rotation = anchor[6]

        bounding_box = np.array([
            [-l/2, -l/2, l/2, l/2],
            [w/2, -w/2, -w/2, w/2]])

        # re-create 3D bounding box in velodyne frame
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)]])

        # Repeat [x, y] 4 times
        four_points = np.tile(anchor[:2], (4, 1))

        # Translate the rotated bounding box by the
        # original center position to obtain the final box
        corner_box = np.dot(rotation_matrix, bounding_box) + \
            four_points.transpose()

        # [XXXX, YYYY] -> [XY, XY, XY, XY]
        corner_box = corner_box.transpose()
        anchors_corner[i] = corner_box

    return anchors_corner


def corner_to_standup_box2d(boxes_corner):
    '''
    Convert corner coordinates to xyxy form boxes
    (N, 4, 2) → (N, 4): [x1, y1, x2, y2]

    Parameters:
        boxes_corner (arr): the boxes in corner notation

    Returns:
        arr: the boxes with the max and min x and y values from the input
    '''
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))

    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)  # X1
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)  # Y1
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)  # X2
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)  # Y2

    return standup_boxes2d


def center_to_corner_box2d(boxes_center):
    # (N, 5) -> (N, X, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes3d_corner = box3d_center_to_corner(
        boxes3d_center,
    )

    return boxes3d_corner[:, 0:4, 0:2]


def box3d_corner_to_center_batch(box3d_corner):
    '''
    Convert a batch of ground truth bounding boxes from corner to center
    notation (N, 8, 3) -> (N, 7)

    Parameters:
        box3d_corner (arr): bounding boxes in corner notation

    Returns:
        arr: bounding boxes in xyzhwlr notation
    '''
    assert box3d_corner.ndim == 3

    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    #     2―――――1       2―――――1
    #    / top /|      /| bck |     -> which points the elements of the input
    #   3―――――0 |     3 |     |        array refer to in the bounding box,
    #   |     | 5     | 6―――――5        by index
    #   |     |/      |/ bot /
    #   7―――――4       7―――――4

    # Define height as the difference between the top and bottom coordinates
    height = abs(np.mean(box3d_corner[:, 4:, 2] - box3d_corner[:, :4, 2],
                         axis=1, keepdims=True))
    #                      _______________
    # width = average of  √ (pt₁ - pt₂)²    for all points on the same side
    width = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
             np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4
    #                       _______________
    # length = average of  √ (pt₁ - pt₂)²   for all points on the opposite side
    length = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
              np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
              np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
              np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    # theta = average of atan2(Δy, Δx) for non-diagonals on the top of the bbox
    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, height, width, length, theta], axis=1).reshape(
        box3d_corner.shape[0], 7)


def delta_to_boxes3d(deltas, anchors):
    '''
    Convert regression map deltas to bounding boxes

    Parameters:
        deltas (arr): (N, W, L, 14): the regression output,
            where N = batch size
        anchors (arr): (X, 7): the anchors, where X = number of anchors

    ReturnsL
        arr: the bounding boxes
    '''
    batch_size = deltas.shape[0]

    # View in the same form as a list of all anchors
    deltas = deltas.view(batch_size, -1, 7)

    anchors = torch.FloatTensor(anchors)
    boxes3d = torch.zeros_like(deltas)

    if deltas.is_cuda:
        anchors = anchors.cuda()
        boxes3d = boxes3d.cuda()

    anchors_reshaped = anchors.view(-1, 7)
    #       _______________
    # dᵃ = √ (lᵃ)² + (wᵃ)²      is the diagonal of the base
    #                           of the anchor box (See 2.2)
    anchors_diagonal = torch.sqrt(
        anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)

    # Repeat across the batch and across [X, Y]
    anchors_diagonal = anchors_diagonal.repeat(
        batch_size, 2, 1).transpose(1, 2)

    # Repeat over the batch size
    anchors_reshaped = anchors_reshaped.repeat(
        batch_size, 1, 1)

    # Δx = (xᵍ - xᵃ) / dᵃ and Δy = (yᵍ - yᵃ) / dᵃ
    # so x = (Δx * dᵃ) + xᵃ and y = (Δy * dᵃ) + yᵃ
    boxes3d[..., [0, 1]] = torch.mul(
        deltas[..., [0, 1]], anchors_diagonal) \
        + anchors_reshaped[..., [0, 1]]

    # Δz = (zᵍ - zᵃ) / hᵃ so z = (Δz * hᵃ) + zᵃ
    boxes3d[..., [2]] = torch.mul(
        deltas[..., [2]], anchors_reshaped[..., [3]]) \
        + anchors_reshaped[..., [2]]

    # Δw = log(wᵍ / wᵃ) so w = e^(Δw) * wᵃ
    boxes3d[..., [3, 4, 5]] = torch.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

    # Δ𝜃 = 𝜃ᵍ - 𝜃ᵃ, so 𝜃 = Δ𝜃 + 𝜃ᵃ
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d


def output_to_boxes(prob_score_map, reg_map):
    '''
    Convert VoxelNet output to bounding boxes for visualization

    Parameters:
        prob_score_map (arr): (BS, 2, H, W), probability score map
        reg_map (arr): (BS, 14, H, W), regression map (deltas)

    Returns:
        arr: boxes in center notation
        arr: boxes in corner notation
        arr: scores for the boxes
    '''
    config = load_config()
    batch_size, _, _, _ = prob_score_map.shape
    device = prob_score_map.device

    # Convert regression map back to bounding boxes (center notation)
    batch_boxes3d = delta_to_boxes3d(reg_map, get_anchors())

    batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
    batch_probs = prob_score_map.reshape((batch_size, -1))

    batch_boxes3d = batch_boxes3d.cpu().numpy()
    batch_boxes2d = batch_boxes2d.cpu().numpy()
    batch_probs = batch_probs.cpu().numpy()

    return_box3d = []
    return_score = []
    for batch_id in range(batch_size):
        # Remove boxes under the threshold
        ind = np.where(
            batch_probs[batch_id, :] >= config['nms_score_threshold'])
        tmp_boxes3d = batch_boxes3d[batch_id, ind, ...].squeeze()
        tmp_boxes2d = batch_boxes2d[batch_id, ind, ...].squeeze()
        tmp_scores = batch_probs[batch_id, ind].squeeze()

        # Convert center notation 3d boxes to corner notation 2d boxes
        corner_box2d = center_to_corner_box2d(tmp_boxes2d)

        # Convert from xxyy to xyxy
        boxes2d = corner_to_standup_box2d(corner_box2d)

        # Apply NMS to get rid of duplicates
        ind, cnt = nms(
            torch.from_numpy(boxes2d).to(device),
            torch.from_numpy(tmp_scores).to(device),
            config['nms_threshold'],
            20,
        )
        try:
            ind = ind[:cnt].cpu().detach().numpy()
        except IndexError:
            print('Unable to select NMS-detected boxes, returning None')
            return None, None

        tmp_boxes3d = tmp_boxes3d[ind, ...]
        tmp_scores = tmp_scores[ind]
        return_box3d.append(tmp_boxes3d)
        return_score.append(tmp_scores)

    return_box3d = np.array(return_box3d)

    # Convert center notation 3d boxes to corner notation 3d boxes
    ret_box3d_corner = box3d_center_to_corner_batch(return_box3d)

    return ret_box3d_corner, return_box3d
