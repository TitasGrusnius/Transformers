"""
COCO datasets which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os.path
from PIL import Image
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List


import tensorflow as tf
import tensorflow_datasets as tfds
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import tensorflow_addons as tfa
import datasets.transforms as T

# READ: unclear if this is from PyTorch or not
# import datasets.transforms as T

FIXED_IMAGE_SIZE_W = 400 
FIXED_IMAGE_SIZE_H = 650 

class CocoDetection(tfds.object_detection.Coco):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        # previous:
        # see https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html#CocoDetection.__getitem__
        # super(CocoDetection, self).__init__(img_folder, ann_file)
        # current:
        self.root=img_folder
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        super(CocoDetection, self).__init__()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        img = self._load_image(self.ids[idx])
        target = self._load_target(self.ids[idx])
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target['boxes'], target['labels']


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = tf.convert_to_tensor(mask, dtype=tf.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = tf.stack(masks, dim=0)
    else:
        masks = tf.zeros((0, height, width), dtype=tf.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = tf.constant([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = tf.reshape(tf.convert_to_tensor(boxes, dtype=tf.float32), (-1, 4))
        # convert to numpy to be able to assign to it; boxes[:, 2:] += boxes[:, :2]
        boxes_np = boxes.numpy()
        boxes_np[:, 2:] += boxes_np[:, :2]
        # convert back to tensor
        boxes = tf.convert_to_tensor(boxes_np, dtype=tf.float32)

        boxes_np = boxes.numpy()
        #boxes[:, 0::2].clamp_(min=0, max=w)
        boxes_np[:, 0::2] = tf.clip_by_value(boxes[:, 0::2], clip_value_min=0, clip_value_max=w)
        #boxes[:, 1::2].clamp_(min=0, max=h)
        boxes_np[:, 1::2] = tf.clip_by_value(boxes[:, 1::2], clip_value_min=0, clip_value_max=h)
        boxes = tf.convert_to_tensor(boxes_np, dtype=tf.float32)


        classes = [obj["category_id"] for obj in anno]
        classes = tf.constant(classes, dtype=tf.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = tf.convert_to_tensor(keypoints, dtype=tf.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = tf.convert_to_tensor([obj["area"] for obj in anno])
        iscrowd = tf.convert_to_tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = tf.convert_to_tensor([int(h), int(w)])
        target["size"] = tf.convert_to_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            T.Resize((FIXED_IMAGE_SIZE_W, FIXED_IMAGE_SIZE_H)),   # fixed-image size 
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset