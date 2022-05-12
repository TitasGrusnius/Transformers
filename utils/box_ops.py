# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
from typing import Tuple

import tensorflow as tf

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = tf.unstack(x, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return tf.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = tf.unstack(x, axis=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return tf.stack(b, axis=-1)


def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.dtype in (tf.float32, tf.float64, tf.float16):
        return t if t.dtype in (tf.float32, tf.float64) else tf.cast(t, tf.float)
    else:
        return t if t.dtype in (tf.int32, tf.int64) else tf.cast(t, tf.int)

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = tf.math.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = tf.math.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = tf.nn.relu((rb - lt))   # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).numpy().all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).numpy().all()
    iou, union = box_iou(boxes1, boxes2)

    lt = tf.math.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = tf.math.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh =  tf.nn.relu((rb - lt))  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def merge(box_a: tf.Tensor, box_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Merged two set of boxes so that operations ca be run to compare them
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        Return the two same tensor tiled: (a, b, 4)
    """
    A = tf.shape(box_a)[0] # Number of bbox in box_a
    B = tf.shape(box_b)[0] # Number of bbox in box b
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a = tf.tile(tf.expand_dims(box_a, axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b = tf.tile(tf.expand_dims(box_b, axis=0), [A, 1, 1])
    return tiled_box_a, tiled_box_b