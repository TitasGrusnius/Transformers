import sys
import tensorflow as tf 

sys.path.insert(1, '/Users/ma/Documents/Brown/SP22/Deep_Learning/Transformers/')

import unittest

from model.transformer import TransformerDecoderLayer as TFTransformerDecoderLayer
from golden.transformer import TransformerDecoderLayer as PYTransformerDecoderLayer

class TestDecoder(unittest.TestCase):

    def test_decoder_layer(self):
        tf_decoder_layer = TFTransformerDecoderLayer(d_model=2, nhead=2, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False) 
        py_decoder_layer = PYTransformerDecoderLayer(d_model=2, nhead=2, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False) 

        tgt = tf.random.uniform(shape=[4, 16])
        memory = tf.random.uniform(shape=[4, 16])
       
        tgt_mask = tf.random.uniform(shape=[8, 2])
        memory_mask = tf.random.uniform(shape=[4, 16])
        tgt_key_padding_mask = tf.random.uniform(shape=[8, 16])
        memory_key_padding_mask = tf.random.uniform(shape=[4, 16])
        query_pos = tf.random.uniform(shape=[8, 16])
        pos = tf.random.uniform(shape=[8, 16])

        layer =  tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)
        tf_decoder_layer.self_attn(tgt, memory)


        # tf_decoder_layer.forward(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, query_pos=query_pos, pos=pos,  training=True)
        # py_decoder_layer.forward(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, training=True)

    def test_decoder(self):
        pass 



if __name__ == '__main__':
    unittest.main()