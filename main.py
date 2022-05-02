import tensorflow as tf
import argparse
from pathlib import Path
from datasets import build_dataset, get_coco_api_from_dataset
import tensorflow_datasets as tfds
from random import sample, shuffle
from datasets.transforms import pad_labels

import PIL
import matplotlib.pyplot as plt
import utils 

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

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
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

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

    # batch dataset
    train_data = train_data.batch(args.batch_size, drop_remainder=True)

    #### TRYING TO GET BATCHING TO WORK WITH VARIABLE-SIZED IMAGES BUT NO AVAIL
    #train_data = train_data.padded_batch(args.batch_size, padded_shapes=([[None], [None], [None]],[],[]), drop_remainder=True)
    #train_data = train_data.map(lambda imgs, boxes, labels: tf.numpy_function(utils.misc.collate_fn, [imgs, boxes, labels], [tf.float32, tf.float32, tf.int64]))

    # TODO: Display First 3 images in training data with their boxes
    for epoch, (img, box, label) in enumerate(train_data): 
        image = tf.keras.preprocessing.image.array_to_img(img[0])
        image.show()
        break 
    
    #### Training Loop #### 



    #### Testing Loop #### 

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)