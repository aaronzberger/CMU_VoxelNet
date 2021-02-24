import torch

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = torch.zeros_like(scores).long()
    count = 0
    if torch.numel(boxes) == 0 or torch.numel(scores) == 0:
        return keep, count

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    # print(boxes.shape, boxes[0], scores.shape, scores[0])
    sorted, idx = torch.sort(scores, dim=0)  # sort in ascending order
    # print(sorted.shape, idx.shape)

    # I = I[v >= 0.01]
    try:
        idx = idx[-top_k:]  # indices of the top-k largest boxes
    except IndexError:
        print("Unable to select the {} top values from these indices".format(top_k))
        return keep, count
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    while torch.numel(idx) > 0:
        i = idx[-1]  # index of current highest score

        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove this element from the indices list

        # load bboxes of next highest vals
        torch.index_select(x1, dim=0, index=idx, out=xx1)
        torch.index_select(y1, dim=0, index=idx, out=yy1)
        torch.index_select(x2, dim=0, index=idx, out=xx2)
        torch.index_select(y2, dim=0, index=idx, out=yy2)

        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1

        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h

        # IoU = i / (area(a) + area(b) - i)
        # load remaining areas
        rem_areas = torch.index_select(area, dim=0, index=idx)
        union = (rem_areas - inter) + area[i]

        IoU = inter / union  # store result in iou

        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    return keep, count
