"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import tensorflow as tf 

class Transformer(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = tf.keras.layers.LayerNormalization(1e-05) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = tf.keras.layers.LayerNormalization(1e-05)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        # Uniform Xavier 
        initializer = tf.keras.initializers.GlorotUniform()
        for p in self.trainable_variables():
            if p.dim() > 1:
                p = initializer(shape=p.shape)
                
        #------------------------Code Difference: Manar's Guess----------------------
        #for p in self.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_uniform_(p)


    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = tf.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

    
class TransformerEncoder(tf.keras.Model):
    def __init__(self, encoder_layer, num_layers, norm=None):       
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers     
        self.norm = norm
        
    def forward(self, src,
                mask = None,
                src_key_padding_mask = None,
                pos = None):
        
        output = src
        for layer in self.layers:           
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
        if self.norm is not None:
            output = self.norm(output)

        return output
    

class TransformerDecoder(tf.keras.Model):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None, training=False):
       
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, training=training)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return tf.stack(intermediate)

        return output.unsqueeze(0)
    
    
class TransformerEncoderLayer(tf.keras.Model):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        
        super().__init__()
        self.self_attn = tf.keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = tf.keras.layers.Dense(units=dim_feedforward)
        self.dropout = f.keras.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(units=d_model)
        
        self.norm1 = tf.keras.layers.LayerNormalization(1e-05)
        self.norm2 = tf.keras.layers.LayerNormalization(1e-05)
        
        self.dropout1 = tf.keras.Dropout(dropout)
        self.dropout2 = tf.keras.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    
    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src
    
    
    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    

class TransformerDecoderLayer(tf.keras.Model):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.self_attn = tf.keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=nhead, dropout=dropout, name='D_self_atten')
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=nhead, dropout=dropout, name='D_multi_head_atten')

        # Implementation of Feedforward model
        self.linear1 = tf.keras.layers.Dense(units=dim_feedforward)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(units=d_model)

        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05)  # epsilon default is different between tensroflow and pytorch
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05)
        self.norm3 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask= None,
                    tgt_key_padding_mask= None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None, 
                    training=False):
        
        q = k = self.with_pos_embed(tgt, query_pos)
        
        # TODO: need to figure out how to implement  key_padding_mask=tgt_key_padding_mask
        tgt2 = self.self_attn(query=q, key=k, value=tgt, attention_mask=tgt_mask)

        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt = self.norm1(tgt)
        
        # TODO: need to figure out how to implement key_padding_mask=memory_key_padding_mask
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attention_mask=memory_mask)
       
        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2,  training=training)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask= None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None,
                    training=False):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        # TODO: need to figure out how to implement  key_padding_mask=tgt_key_padding_mask
        tgt2 = self.self_attn(query=q, key=k, value=tgt2, attention_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt2 = self.norm2(tgt)
        
        # TODO: need to figure out how to implement key_padding_mask=memory_key_padding_mask
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attention_mask=memory_mask)

        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2, training=training)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask= None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos= None,
                query_pos = None, training=False):

        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, training)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, training)



def _get_clones(module, N):
    return [copy.deepcopy(module) for i in range(N)]

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return tf.keras.activations.relu
    if activation == "gelu":
        return tf.keras.activations.gelu
    if activation == "glu":
        return tf.keras.activations.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")