import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
from datasets import build_dataset, get_coco_api_from_dataset
import tensorflow_datasets as tfds
from random import sample, shuffle
from datasets.transforms import pad_labels
from datasets.coco import COCO_CLASS_NAME

from model.detr import build 
from engine import train_one_epoch 

import PIL
import matplotlib.pyplot as plt
from utils.plot import numpy_bbox_to_image

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    return parser


def main(args):
    #### Data Pipeline #### 
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    train_data = tf.data.Dataset.from_tensor_slices(dataset_train.ids)
    valid_data = tf.data.Dataset.from_tensor_slices(dataset_val.ids)
  
    train_data = train_data.shuffle(1000)
    
    train_data = train_data.map(lambda idx: tf.numpy_function(dataset_train.__getitem__, [idx], [tf.float32, tf.float32, tf.int64]))
    valid_data = valid_data.map(lambda idx: tf.numpy_function(dataset_val.__getitem__, [idx], [tf.float32, tf.float32, tf.int64]))

    # pad images, boxes, and labels
    train_data = train_data.map(lambda imgs, boxes, labels: tf.numpy_function(pad_labels, [imgs, boxes, labels], [tf.float32, tf.float32, tf.int64]))
    valid_data = valid_data.map(lambda imgs, boxes, labels: tf.numpy_function(pad_labels, [imgs, boxes, labels], [tf.float32, tf.float32, tf.int64]))

    # batch dataset
    train_data = train_data.batch(args.batch_size, drop_remainder=True)
    valid_data = valid_data.batch(args.batch_size, drop_remainder=True)

    #### TRYING TO GET BATCHING TO WORK WITH VARIABLE-SIZED IMAGES BUT NO AVAIL
    #train_data = train_data.padded_batch(args.batch_size, padded_shapes=([[None], [None], [None]],[],[]), drop_remainder=True)
    #train_data = train_data.map(lambda imgs, boxes, labels: tf.numpy_function(utils.misc.collate_fn, [imgs, boxes, labels], [tf.float32, tf.float32, tf.int64]))

    # TODO: Display First 3 images in training data with their boxes
    for epoch, (img, box, label) in enumerate(train_data): 
        # image = numpy_bbox_to_image(np.array(img[0]), np.array(box[0]), np.array(label[0]), class_name=COCO_CLASS_NAME)
        # plt.imshow(image)
        # plt.show()
        break 
    
    #### Training Loop #### 
    # Optimizers
    backbone_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_backbone, clipnorm=args.clip_max_norm)
    transformers_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=args.clip_max_norm)
    fnn_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=args.clip_max_norm)

    optimizers = {
        'backbone': backbone_optimizer,
        'transformer': transformers_optimizer,
        'fnn': fnn_optimizer}

    model, criterion, postprocess = build(args)
    for epoch, (img, box, label) in enumerate(train_data): 
        train_one_epoch(model, criterion, optimizers, img, box, label)
        break

    #### Testing Loop #### 

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)