import cv2
import numpy as np 
from matplotlib import patches, text, patheffects
import matplotlib.pyplot as plt

CLASS_COLOR_MAP = np.random.randint(0, 255, (100, 3))

def np_xcycwh_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xy = np.concatenate([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    return bbox_xy

def np_rescale_bbox_xcycwh(bbox_xcycwh: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[xc, yc, w, h], ...]
        @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh) # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled

def numpy_bbox_to_image(image, bbox_list, labels=None, scores=None, class_name=[], config=None):
    """ Numpy function used to display the bbox (target or prediction)
    """
    # rescale image 
    num_classes = 0 
    for l in labels:
            if l!=0: num_classes +=1

    labels = labels[0:num_classes+1]
    bbox_list = bbox_list[0:num_classes+1, :]

    image = ((np.array(image) - np.array(image).min()) * (1/(np.array(image).max() - np.array(image).min()) * 255)).astype('uint8')
    
    bbox_xcycwh = np_rescale_bbox_xcycwh(bbox_list, (image.shape[0], image.shape[1])) 
    bbox_x1y1x2y2 = np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)

    # Set the labels if not defined
    if labels is None: labels = np.zeros((bbox_x1y1x2y2.shape[0]))

    bbox_area = []
    # Go through each bbox
    for b in range(0, bbox_x1y1x2y2.shape[0]):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2-x1)*(y2-y1))

    # Go through each bbox
    for b in np.argsort(bbox_area)[::-1]:
        # Take a new color at reandon for this instance
        instance_color = np.random.randint(0, 255, (3))
        

        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)

        print(x1, y1, x2, y2)
        # Select the class associated with this bbox
        class_id = labels[int(b)]

        if scores is not None and len(scores) > 0:
            label_name = class_name[int(class_id)]   
            label_name = "%s:%.2f" % (label_name, scores[b])
        else:
            label_name = class_name[int(class_id)]    

        class_color = CLASS_COLOR_MAP[int(class_id)]
    
        color = instance_color
        
        multiplier = image.shape[0] / 500
        cv2.rectangle(image, (x1, y1), (x1 + int(multiplier*15)*len(label_name), y1 + 20), class_color.tolist(), -10)
        cv2.putText(image, label_name, (x1+2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * multiplier, (0, 0, 0), 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(class_color.tolist()), 2)

    return image

