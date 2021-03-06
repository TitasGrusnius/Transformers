# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import sys
sys.path.insert(1, '/Users/ma/Documents/Brown/SP22/Deep_Learning/Transformers/')

import numpy as np
import tensorflow as tf 
from typing import Tuple
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, merge


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
        cost_class = -tf.gather(out_prob, tgt_ids,  batch_dims=1, axis=1)
        
        # Compute the L1 cost between boxes
        _p_bbox, _t_bbox = merge(out_bbox, tgt_bbox)
        cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C =  tf.reshape(C, (bs, num_queries, -1))

        sizes = [len(v) for v in boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(tf.split(C, sizes, axis=-1))]
        return [(tf.convert_to_tensor(tf.cast(i, tf.int64), dtype=tf.int64), tf.convert_to_tensor(tf.cast(j, dtype=tf.int64), dtype=tf.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)