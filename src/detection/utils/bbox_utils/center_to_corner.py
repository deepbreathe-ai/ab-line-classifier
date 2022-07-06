import numpy as np


def center_to_corner(boxes):
    '''
    Convert a set of bounding boxes from center format (cx, cy, width, height) to corner format (xmin, ymin, xmax, ymax)
    :param boxes: NumPy array of boxes in center format
    :return: NumPy array of boxes in corner format
    '''
    temp = boxes.copy()
    temp[..., 0] = boxes[..., 0] - (boxes[..., 2] / 2)  # xmin
    temp[..., 1] = boxes[..., 1] - (boxes[..., 3] / 2)  # ymin
    temp[..., 2] = boxes[..., 0] + (boxes[..., 2] / 2)  # xmax
    temp[..., 3] = boxes[..., 1] + (boxes[..., 3] / 2)  # ymax
    return temp
