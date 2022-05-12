"""
Various positional encodings for the transformer.
"""
import sys
sys.path.insert(1, '/Users/ma/Documents/Brown/SP22/Deep_Learning/Transformers/')

import numpy as np
import tensorflow as tf
from utils.misc import NestedTensor


class PositionEmbeddingSine(tf.keras.Model):

    def __init__(self, num_pos_features=64, temperature=10000,
                 normalize=False, scale=None, eps=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps


    def call(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = tf.cast(~tf.cast(mask, tf.int32), tf.float32)
        y_embed = tf.math.cumsum(not_mask, axis=1)
        x_embed = tf.math.cumsum(not_mask, axis=2)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = tf.range(self.num_pos_features, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_features)

        pos_x = x_embed[..., tf.newaxis] / dim_t
        pos_y = y_embed[..., tf.newaxis] / dim_t

        pos_x = tf.stack([tf.math.sin(pos_x[..., 0::2]),
                          tf.math.cos(pos_x[..., 1::2])], axis=4)

        pos_y = tf.stack([tf.math.sin(pos_y[..., 0::2]),
                          tf.math.cos(pos_y[..., 1::2])], axis=4)
        

        shape = [tf.shape(pos_x)[i] for i in range(3)] + [-1]
        pos_x = tf.reshape(pos_x, shape)
        pos_y = tf.reshape(pos_y, shape)

        pos_emb = tf.concat([pos_y, pos_x], axis=3)
        return pos_emb


class PositionEmbeddingLearned(tf.keras.Model):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = tf.keras.layers.Embedding(50, num_pos_feats)
        self.col_embed = tf.keras.layers.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        self.row_embed = tf.random.normal(self.row_embed.shape)
        self.col_embed = tf.random.normal(self.col_embed.shape)

    def call(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        # again has no attribute device
        i = tf.range(w)
        j = tf.range(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = tf.concat([
            x_emb.expand_dims(0).repeat(h, 1, 1),
            y_emb.expand_dims(1).repeat(1, w, 1),
            # (a, b, c, d) represent axes therefore first axis is 0 where it's repeated x.shape[0] times
        ], axis=-1).transpose(perm=[2, 0, 1]).expand_dims(0).repeat(repeats=x.shape[0], axis=0)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    return position_embedding
