# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import sys
import numpy as np
from matplotlib.pyplot import axis
sys.path.insert(1, '/Users/ma/Documents/Brown/SP22/Deep_Learning/Transformers/')

from typing import Union,Dict,Tuple

from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

import tensorflow as tf 

def bbox_xcycwh_to_x1y1x2y2(bbox_xcycwh: np.array):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[xc, yc, w, h], ...]
        @img_size (height, width)
    """
    bbox_x1y1x2y2 = np.zeros_like((bbox_xcycwh))
    bbox_x1y1x2y2[:,0] = bbox_xcycwh[:,0] - (bbox_xcycwh[:,2] / 2)
    bbox_x1y1x2y2[:,2] = bbox_xcycwh[:,0] + (bbox_xcycwh[:,2] / 2)
    bbox_x1y1x2y2[:,1] = bbox_xcycwh[:,1] - (bbox_xcycwh[:,3] / 2)
    bbox_x1y1x2y2[:,3] = bbox_xcycwh[:,1] + (bbox_xcycwh[:,3] / 2)
    bbox_x1y1x2y2 = bbox_x1y1x2y2.astype(np.int32)
    return bbox_x1y1x2y2


def intersect(box_a: tf.Tensor, box_b: tf.Tensor) -> tf.Tensor:
    """
    Compute the intersection area between two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The intersection area [a, b] between each bbox. zero if no intersection
    """
    # resize both tensors to [A,B,2] with the tile function to compare
    # each bbox with the anchors:
    # [A,2] -> [A,1,2] -> [A,B,2]
    # [B,2] -> [1,B,2] -> [A,B,2]
    # Then we compute the area of intersect between box_a and box_b.
    # box_a: (tensor) bounding boxes, Shape: [n, A, 4].
    # box_b: (tensor) bounding boxes, Shape: [n, B, 4].
    # Return: (tensor) intersection area, Shape: [n,A,B].

    A = tf.shape(box_a)[0] # Number of possible bbox
    B = tf.shape(box_b)[0] # Number of anchors

    #print(A, B, box_a.shape, box_b.shape)
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymax = tf.tile(tf.expand_dims(box_a[:, 2:], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymax = tf.tile(tf.expand_dims(box_b[:, 2:], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    above_right_corner = tf.math.minimum(tiled_box_a_xymax, tiled_box_b_xymax)


    # Upper Left Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymin = tf.tile(tf.expand_dims(box_a[:, :2], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymin = tf.tile(tf.expand_dims(box_b[:, :2], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    upper_left_corner = tf.math.maximum(tiled_box_a_xymin, tiled_box_b_xymin)


    # If there is some intersection, both must be > 0
    inter = tf.nn.relu(above_right_corner - upper_left_corner)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter


def jaccard(box_a: tf.Tensor, box_b: tf.Tensor, return_union=False) -> tf.Tensor:
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The Jaccard overlap [a, b] between each bbox
    """
    # Get the intersectin area
    inter = intersect(box_a, box_b)

    # Compute the A area
    # (xmax - xmin) * (ymax - ymin)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    # Tile the area to match the anchors area
    area_a = tf.tile(tf.expand_dims(area_a, axis=-1), [1, tf.shape(inter)[-1]])

    # Compute the B area
    # (xmax - xmin) * (ymax - ymin)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    # Tile the area to match the gt areas
    area_b = tf.tile(tf.expand_dims(area_b, axis=-2), [tf.shape(inter)[-2], 1])

    union = area_a + area_b - inter

    if return_union is False:
        # Return the intesect over union
        return inter / union
    else:
        return inter / union, union

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

def xy_min_xy_max_to_yx_min_yx_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return tf.concat([
        bbox[:,1:2],
        bbox[:,0:1],
        bbox[:,3:4],
        bbox[:,2:3]
    ], axis=-1)

def yx_min_yx_max_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return tf.concat([
        bbox[:,1:2],
        bbox[:,0:1],
        bbox[:,3:4],
        bbox[:,2:3]
    ], axis=-1)


def xy_min_xy_max_to_xcycwh(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [xc, yc, w, h]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h]
    bbox_xcycwh = tf.concat([bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]], axis=-1)
    return bbox_xcycwh



def xcycwh_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    print(bbox.shape)
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xyxy = tf.concat([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    # Be sure to keep the values btw 0 and 1
    bbox_xyxy = tf.clip_by_value(bbox_xyxy, 0.0, 1.0)
    return bbox_xyxy


def xcycwh_to_yx_min_yx_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [ymin, xmin, ymax, xmax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    bbox = xcycwh_to_xy_min_xy_max(bbox)
    bbox = xy_min_xy_max_to_yx_min_yx_max(bbox)
    return bbox


def yx_min_yx_max_to_xcycwh(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xc, yc, w, h]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    bbox = yx_min_yx_max_to_xy_min_xy_max(bbox)
    bbox = xy_min_xy_max_to_xcycwh(bbox)
    return bbox



"""
Numpy Transformations
"""

def xy_min_xy_max_to_xcycwh(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [xc, yc, w, h]
    Args:
        bbox A (np.array) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h]
    bbox_xcycwh = np.concatenate([bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]], axis=-1)
    return bbox_xcycwh


def np_xcycwh_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xy = np.concatenate([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    return bbox_xy



def np_yx_min_yx_max_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (np.array) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return np.concatenate([
        bbox[:,1:2],
        bbox[:,0:1],
        bbox[:,3:4],
        bbox[:,2:3]
    ], axis=-1)



def np_rescale_bbox_xcycwh(bbox_xcycwh: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[xc, yc, w, h], ...]
        @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh) # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_yx_min_yx_max(bbox_xcycwh: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[y_min, x_min, y_max, x_max], ...]
        @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh) # Be sure to work with a numpy array
    scale = np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_xy_min_xy_max(bbox: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox: [[x_min, y_min, x_max, y_max], ...]
        @img_size (height, width)
    """
    bbox = np.array(bbox) # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_rescaled = bbox * scale
    return bbox_rescaled

def np_tf_linear_sum_assignment(matrix):

    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]

    #print(matrix.shape, target_indices, pred_indices)

    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)

    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)

    #print('target_indices', target_indices)
    #print("pred_indices", pred_indices)

    return [target_indices, pred_indices, target_selector, pred_selector]

class HungarianMatcher(tf.keras.Model):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    #@tf.no_gradient()
    def call(self, outputs, boxes, labels):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        bs, num_queries, num_classes =  outputs["pred_logits"].shape 

        # We flatten to compute the cost matrices in a batch
        out_prob = tf.nn.softmax(tf.reshape(outputs["pred_logits"], (bs*num_queries, num_classes)), axis=-1)  # [batch_size * num_queries, num_classes]
        out_bbox = tf.reshape(outputs["pred_boxes"], (bs*num_queries, 4))  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = tf.concat([v for v in labels], axis=0)
        tgt_bbox = tf.concat([v for v in boxes], axis=0)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = tf.stop_gradient(-tf.gather(out_prob, tgt_ids,  batch_dims=1, axis=1))
        
        # Compute the L1 cost between boxes
        _p_bbox, _t_bbox = merge(out_bbox, tgt_bbox)
        cost_bbox =  tf.stop_gradient(tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1))

        # Compute the giou cost betwen boxes
        cost_giou = tf.stop_gradient(-generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)))

        # Final cost matrix
        C = tf.stop_gradient(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
        C =  tf.stop_gradient(tf.reshape(C, (bs, num_queries, -1)))

        sizes = tf.stop_gradient([len(v) for v in boxes])
        indices = tf.stop_gradient([linear_sum_assignment(c[i]) for i, c in enumerate(tf.split(C, sizes, axis=-1))])
        return [(tf.convert_to_tensor(tf.cast(i, tf.int64), dtype=tf.int64), tf.convert_to_tensor(tf.cast(j, dtype=tf.int64), dtype=tf.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)