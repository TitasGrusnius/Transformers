"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import tensorflow as tf 

class Transformer(tf.keras.Model):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__(kwargs)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                 dropout, activation, normalize_before)
        encoder_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
                                     
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead


    def call(self, src, mask, query_embed, pos_embed, training=False):

        bs, h, w, c = src.shape
        src = tf.reshape(src, [bs, -1, self.d_model])
        src = tf.transpose(src, [1, 0, 2])

        pos_embed = tf.reshape(pos_embed, [bs, -1, self.d_model])
        pos_embed = tf.transpose(pos_embed, [1, 0, 2])

        query_embed = tf.expand_dims(query_embed, axis=1)
        query_embed = tf.tile(query_embed, [1, bs, 1])

        mask = tf.reshape(mask, [bs, -1])

        tgt = tf.zeros_like(query_embed)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, training=training)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed, training=training)

        return  tf.transpose(hs, [0, 2, 1, 3]), tf.reshape(tf.transpose(memory, [1, 0, 2]), [bs, h, w, self.d_model])


class TransformerEncoder(tf.keras.Model):
    def __init__(self, encoder_layer, num_layers, norm=None):       
        super().__init__()
        self.t_layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers     
        self.norm = norm
        
    def call(self, src,
                mask = None,
                src_key_padding_mask = None,
                pos = None, training=False):
        
        output = src
        for layer in self.t_layers:           
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, training=training)
            
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(tf.keras.Model):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, **kwargs):
        super().__init__(kwargs)
        self.t_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def call(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None, training=False):
       
        output = tgt
        intermediate = []

        for layer in self.t_layers:
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


class TransformerEncoderLayer(tf.keras.layers.Layer):
   
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        
        super().__init__(**kwargs)
        self.self_attn = tf.keras.layers.MultiHeadAttention(d_model, nhead, dropout=dropout, name='self_attn')

        # Implementation of Feedforward model
        self.linear1 = tf.keras.layers.Dense(units=dim_feedforward)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear2 = tf.keras.layers.Dense(units=d_model)
        
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05)
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None, training=False):
        
        q = k = self.with_pos_embed(src, pos)
        # TODO:  key_padding_mask=src_key_padding_mask
        src2 = self.self_attn(query=q, key=k, value=src, attention_mask=src_mask)
        src = src + self.dropout1(src2, training=training)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)), training=training))
        src = src + self.dropout2(src2, training=training)
        src = self.norm2(src)
        
        return src
    
    
    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None, training=False):
        
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        
        # TODO: key_padding_mask=src_key_padding_mask
        src2 = self.self_attn(query=q, key=k, value=src, attention_mask=src_mask)
        
        src = src + self.dropout1(src2, training=training)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2, training=training)
        
        return src
    
    
    def call(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None, training=False):
        
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, training=training)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, training=training)


class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__(kwargs)

        self.self_attn = tf.keras.layers.MultiHeadAttention(d_model, nhead, dropout=dropout, name='self_attn')

        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, name='multihead_attn')
      
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
        tgt2 = self.multihead_attn((self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory), attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=False)

        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)), training=training))
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
        tgt2 = self.self_attn((q, k, tgt2), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)
        tgt = tgt + self.dropout1(tgt2, training=training)
        tgt2 = self.norm2(tgt)
        
        # TODO: need to figure out how to implement key_padding_mask=memory_key_padding_mask
        tgt2 = self.multihead_attn((self.with_pos_embed(tgt2, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory), attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2, training=training)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2)), training=training))
        tgt = tgt + self.dropout3(tgt2, training=training)
        return tgt

    def call(self, tgt, memory,
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



### SOURCE: https://github.com/Visual-Behavior/detr-tensorflow/blob/main/detr_tf/networks/transformer.py#L237
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        

    def build(self, input_shapes):
        in_dim = sum([shape[-1] for shape in input_shapes[:3]])

        self.in_proj_weight = self.add_weight(
            name='in_proj_kernel', shape=(in_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True )
        self.in_proj_bias = self.add_weight(
            name='in_proj_bias', shape=(in_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_weight = self.add_weight(
            name='out_proj_kernel', shape=(self.model_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_bias = self.add_weight(
            name='out_proj_bias', shape=(self.model_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )

    def call(self, inputs, attn_mask=None, key_padding_mask=None,
             need_weights=True, training=False):

        query, key, value = inputs

        batch_size = tf.shape(query)[1]
        target_len = tf.shape(query)[0]
        source_len = tf.shape(key)[0]

        W = self.in_proj_weight[:self.model_dim, :]
        b = self.in_proj_bias[:self.model_dim]

        WQ = tf.matmul(query, W, transpose_b=True) + b

        W = self.in_proj_weight[self.model_dim:2*self.model_dim, :]
        b = self.in_proj_bias[self.model_dim:2*self.model_dim]
        WK = tf.matmul(key, W, transpose_b=True) + b

        W = self.in_proj_weight[2*self.model_dim:, :]
        b = self.in_proj_bias[2*self.model_dim:]
        WV = tf.matmul(value, W, transpose_b=True) + b

        WQ *= float(self.head_dim) ** -0.5
        WQ = tf.reshape(WQ, [target_len, batch_size * self.num_heads, self.head_dim])
        WQ = tf.transpose(WQ, [1, 0, 2])
        
        WK = tf.reshape(WK, [source_len, batch_size * self.num_heads, self.head_dim])
        WK = tf.transpose(WK, [1, 0, 2])

        WV = tf.reshape(WV, [source_len, batch_size * self.num_heads, self.head_dim])
        WV = tf.transpose(WV, [1, 0, 2])
        
        attn_output_weights = tf.matmul(WQ, WK, transpose_b=True)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        """
        if key_padding_mask is not None:
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size, self.num_heads, target_len, source_len])

            key_padding_mask = tf.expand_dims(key_padding_mask, 1)
            key_padding_mask = tf.expand_dims(key_padding_mask, 2)
            key_padding_mask = tf.tile(key_padding_mask, [1, self.num_heads, target_len, 1])

            #print("before attn_output_weights", attn_output_weights.shape)
            attn_output_weights = tf.where(key_padding_mask,
                                           tf.zeros_like(attn_output_weights) + float('-inf'),
                                           attn_output_weights)
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size * self.num_heads, target_len, source_len])
        """


        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=-1)
        attn_output_weights = self.dropout(attn_output_weights, training=training)

        attn_output = tf.matmul(attn_output_weights, WV)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [target_len, batch_size, self.model_dim])
        attn_output = tf.matmul(attn_output, self.out_proj_weight,
                                transpose_b=True) + self.out_proj_bias

        if need_weights:
            attn_output_weights = tf.reshape(attn_output_weights,
                            [batch_size, self.num_heads, target_len, source_len])
            # Retrun the average weight over the heads
            avg_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights
        
        return attn_output


def _get_clones(module, N):
    return [copy.deepcopy(module) for i in range(N)]

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return tf.keras.activations.relu
    if activation == "gelu":
        return tf.keras.activations.gelu
    if activation == "glu":
        return tf.keras.activations.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")