import numpy as np


def corner_to_center(boxes):
    '''
    Convert a set of bounding boxes from corner format (xmin, ymin, xmax, ymax) to center format (cx, cy, width, height)
    :param boxes: NumPy array of boxes in corner format
    :return: NumPy array of boxes in center format
    '''
    temp = boxes.copy()
    width = np.abs(boxes[..., 0] - boxes[..., 2])
    height = np.abs(boxes[..., 1] - boxes[..., 3])
    temp[..., 0] = boxes[..., 0] + (width / 2)  # cx
    temp[..., 1] = boxes[..., 1] + (height / 2)  # cy
    temp[..., 2] = width
    temp[..., 3] = height
    return temp
