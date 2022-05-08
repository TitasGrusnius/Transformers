"""
DETR model and criterion classes.
"""
"""
TJ- These are commented out pytorch imports:
import torch
import torch.nn.functional as F
from torch import nn
"""

import tensorflow as tf

import sys
sys.path.insert(1, '/Users/ma/Documents/Brown/SP22/Deep_Learning/Transformers/')

#TJ-not sure how to deal with all of these imports
from utils import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer

#TJ-Not sure about this subclassing but I think they're equivalent
class DETR(tf.keras.Model):
    """ This is the DETR module that performs object detection """
    #TJ-DONE with caveats
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = tf.keras.layers.Dense(num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = tf.keras.layers.Embedding(num_queries, hidden_dim, embeddings_initializer="uniform")
        self.query_embed.build(input_shape=None)
        self.input_proj = tf.keras.layers.Conv2D(hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    #TJ-DONE 
    def call(self, samples: NestedTensor, training=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, tf.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weights[0], pos[-1], training=training)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = tf.sigmoid(self.bbox_embed(hs))
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    #TJ-NOT DONE, not sure what this decorator should be
    #@torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(tf.keras.Model):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    #TJ-DONE
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.t_losses = losses
        empty_weight = tf.ones(self.num_classes + 1)
        empty_weight_np = empty_weight.numpy()
        empty_weight_np[-1] = self.eos_coef
        empty_weight = tf.convert_to_tensor(empty_weight_np, dtype=tf.float32)
        empty_weight = tf.constant(empty_weight)
       # self.register_buffer('empty_weight', empty_weight)

    #TJ-DONE
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = tf.concat([t["labels"][J] for t, (_, J) in zip(targets, indices)], axis=0)
        target_classes = tf.fill(src_logits.shape[:2], self.num_classes)
        target_classes[idx] = target_classes_o

        loss_ce = tf.nn.softmax_cross_entropy_with_logits(target_classes, src_logits.transpose(1, 2))
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    #TJ-DONE
    #@tf.stop_gradient()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = tf.stop_gradient(outputs['pred_logits'])
        tgt_lengths = tf.stop_gradient(tf.convert_to_tensor([len(v["labels"]) for v in targets]))
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = tf.stop_gradient((pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1))
        card_err = tf.stop_gradient(tf.math.reduce_mean(tf.abs(card_pred.float()-tgt_lengths.float())))
        losses = tf.stop_gradient({'cardinality_error': card_err})
        return losses

    #TJ-DONE
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = tf.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = tf.abs(src_boxes-target_boxes)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - tf.matrix_diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    #TJ-DONE
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = tf.concat([tf.fill(src, i) for i, (src, _) in enumerate(indices)], axis=0)
        src_idx = tf.concat([src for (src, _) in indices], axis=0)
        return batch_idx, src_idx

    #TJ-DONE
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = tf.concat([tf.fill(tgt, i) for i, (_, tgt) in enumerate(indices)], axis=0)
        tgt_idx = tf.concat([tgt for (_, tgt) in indices], axis=0)
        return batch_idx, tgt_idx

    #TJ-DONE (unchanged, I think this is good as is)
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    #TJ-DONE WITH CAVEATS
    def call(self, outputs, boxes, labels):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, boxes, labels)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = tf.convert_to_tensor([num_boxes], dtype=float)
        #TJ- NOT SURE HOW TO DEAL WITH THE NEXT 2 LINES
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = tf.get_static_value(tf.clip_by_value(num_boxes / get_world_size(), clip_value_min=1))

        # Compute all the requested losses
        losses = {}
        for loss in self.t_losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.t_losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

#TJ-DONE
#TJ-Not sure about this subclassing but I think they're equivalent
class PostProcess(tf.keras.Model):
    """ This module converts the model's output into the format expected by the coco api"""
    #TJ-DONE
#    @tf.stop_gradient()
    def call(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = tf.stop_gradient(outputs['pred_logits'], outputs['pred_boxes'])

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = tf.stop_gradient(tf.nn.softmax(out_logits, axis=-1))
        scores, labels = tf.stop_gradient(prob[..., :-1].max(-1))

        # convert to [x0, y0, x1, y1] format
        boxes = tf.stop_gradient(box_ops.box_cxcywh_to_xyxy(out_bbox))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = tf.stop_gradient(target_sizes.unbind(1))
        scale_fct = tf.stop_gradient(tf.stack([img_w, img_h, img_w, img_h], axis=1))
        boxes = tf.stop_gradient(boxes * scale_fct[:, None, :])

        results = tf.stop_gradient([{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)])

        return results


class MLP(tf.keras.Model):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.t_layers = [tf.keras.layers.Dense(k) for n, k in zip([input_dim] + h, h + [output_dim])]

    def call(self, x):
        for i, layer in enumerate(self.t_layers):
            x = tf.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    device = tf.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
  
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    # TODO: buffers to GPU ? 
    #criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
   
    return model, criterion, postprocessors
