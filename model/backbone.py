"""
Backbone modules.
"""
import sys
sys.path.insert(1, '/Users/ma/Documents/Brown/SP22/Deep_Learning/Transformers/')

import tensorflow as tf
from typing import Dict, List

from utils.misc import NestedTensor
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(tf.keras.Model):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight = tf.constant(tf.ones(n))
        self.bias = tf.constant(tf.zeros(n))
        self.running_mean = tf.constant(tf.zeros(n))
        self.running_var = tf.constant(tf.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def call(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        # tf.constant
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(tf.keras.Model):

    def __init__(self, backbone: tf.keras.Model, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # TODO: there is not equivalent to requires_grad 
        for layer in backbone.layers:
            if not train_backbone or layer.name.startswith('bn'): # or 'layer2' not in layer.name and 'layer3' not in layer.name and 'layer4' not in layer.name
                layer.trainable = False # see https://stackoverflow.com/questions/67885869/how-to-freeze-batch-norm-layers-during-transfer-learning
                #.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        # self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # Titas guess for below: I couldn't find a direct version of IntermediateLayerGetter in Tensorflow, but searched
        # online and found that model.get_layer(layer_name).outputs returns intermediate layers
        # self.body = []
        # for layer in backbone.layers:
        #     self.body.append(backbone.get_layer(name=layer.name))
        
        self.body = backbone
        self.num_channels = num_channels

    def call(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None
        masks = tf.cast(m, tf.int32)
        masks = tf.expand_dims(masks, -1)
        masks = tf.compat.v1.image.resize_nearest_neighbor(masks, tf.shape(xs)[1:3], align_corners=False, half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        out["layer4"] = NestedTensor(xs, masks)
        return out

    def get_trainable_variables(self):
        list = []
        for layer in self.body.layers:
            if not layer.name.startswith('bn'):
                list.append(layer.trainable_variables)
        return list 

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        
        backbone = getattr(tf.keras.applications, name).ResNet50(include_top=False, weights='imagenet')
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(tf.keras.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__([backbone, position_embedding])

    def call(self, tensor_list: NestedTensor):
        xs = self.layers[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(tf.cast(self.layers[1](x), x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model