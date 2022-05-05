# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import sys
sys.path.insert(1, '/Users/ma/Documents/Brown/SP22/Deep_Learning/Transformers/')

from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

import tensorflow as tf 

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
    def forward(self, outputs, targets):
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

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = tf.concat([v["labels"] for v in targets])
        tgt_bbox = tf.concat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = tf.stop_gradient(-out_prob[:, tgt_ids])

        # Compute the L1 cost between boxes
        cost_bbox = tf.stop_gradient(tf.norm(out_bbox, tgt_bbox, ord='fro'))

        # Compute the giou cost betwen boxes
        cost_giou = tf.stop_gradient(-generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)))

        # Final cost matrix
        C = tf.stop_gradient(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
        C =  tf.stop_gradient(C.view(bs, num_queries, -1).cpu())

        sizes = tf.stop_gradient([len(v["boxes"]) for v in targets])
        indices = tf.stop_gradient([linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))])
        return tf.stop_gradient([(tf.convert_to_tensor(i, dtype=tf.int64), tf.convert_to_tensor(j, dtype=tf.int64)) for i, j in indices])


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)