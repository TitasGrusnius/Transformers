"""
Transforms and data augmentation for both image + bbox.
"""
import PIL
import random
import tensorflow as tf

from utils.box_ops import box_xyxy_to_cxcywh
from utils.misc import interpolate

# Unsure of line 243


def crop(image, target, region):
    # cropped_image = F.crop(image, *region)
    i, j, h, w = region
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height=i, offset_width=j, target_height=h, target_width=w)

    target = target.copy()

    # should we do something wrt the original size?
    target["size"] = tf.constant([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = tf.convert_to_tensor([w, h], dtype=tf.float32)
        cropped_boxes = boxes - tf.convert_to_tensor([j, i, j, i], dtype=tf.float32)
        cropped_boxes = tf.minimum(tf.reshape(cropped_boxes, (-1, 2, 2)), max_size)
        cropped_boxes = tf.clip_by_value(cropped_boxes, clip_value_min=0, clip_value_max=tf.math.reduce_max(cropped_boxes))
        area = tf.math.reduce_prod((cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]), axis=1)
        target["boxes"] = tf.reshape(cropped_boxes,(-1, 4))
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = tf.reshape(target['boxes'], (-1, 2, 2))
            keep = tf.reduce_all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = tf.image.flip_left_right(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes_np = boxes.numpy()
        boxes = boxes_np[:, [2, 1, 0, 3]] * tf.convert_to_tensor([-1, 1, -1, 1]) + tf.convert_to_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    
    if type(image) != PIL.Image.Image:
        image = tf.keras.preprocessing.image.array_to_img(image)

    size = get_size(image.size, size, max_size)
    rescaled_image = tf.image.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.shape, image.size))
    ratio_width, ratio_height = ratios
    
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = tf.cast(boxes, dtype=tf.float32) * tf.convert_to_tensor([ratio_width, ratio_height, ratio_width, ratio_height], dtype=tf.float32)
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = tf.constant([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = tf.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = tf.Tensor(padded_image.size[::-1])
    if "masks" in target:
        # target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
        target['masks'] = tf.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


def pad_labels(images , boxes, labels):
    nb_bbox = boxes.shape[0]

    boxes = tf.pad(boxes, [[0, 100 - nb_bbox], [0, 0]], mode='CONSTANT', constant_values=0)
    labels = tf.reshape(labels, (labels.shape[0], 1))
    labels = tf.pad(labels, [[0, 100 - nb_bbox], [0, 0]], mode='CONSTANT', constant_values=0)

    return images, boxes, labels

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        # region = T.RandomCrop.get_params(img, self.size)
        # need to check this as can't find tensorflow equivalent
        region = tf.image.random_crop(img, self.size).shape
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        if type(img) != PIL.Image.Image:
            img = tf.keras.preprocessing.image.array_to_img(img)
        tw = random.randint(self.min_size, min(img.width, self.max_size))
        th = random.randint(self.min_size, min(img.height, self.max_size))
        # region = T.RandomCrop.get_params(img, [h, w])
        h, w = img.height, img.width

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        region = i,j,th,tw
        #region = tf.image.random_crop(img, [h, w, 3]), tf.image.random_crop(img, [h, w, 3]).shape
        return crop(img, target, region)


class Resize(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        if type(img) != PIL.Image.Image:
            img = tf.keras.preprocessing.image.array_to_img(img)

        return resize(img, target, self.new_size)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return tf.convert_to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        # self.eraser = T.RandomErasing(*args, **kwargs)
        # Not sure about this one
        self.eraser = tf.image.random_saturation(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # image = F.normalize(image, mean=self.mean, std=self.std)
        image -= self.mean
        image /= self.std
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / tf.constant([w, h, w, h], dtype=tf.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string