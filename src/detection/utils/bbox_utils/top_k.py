import tensorflow as tf


def top_k(boxes, k, pad_boxes=False, sort_boxes=True):
    '''
    Perform top-k filtering for set of boxes and optionally pad boxes before filtering
    :param boxes: Tensor of all boxes
    :param k: Number of boxes to keep
    :param pad_boxes: Boolean indicating whether to pad initial set of boxes with zeros
    :param sort_boxes: Boolean indicating whether returned Tensor is sorted or not
    :return: Tensor of top k boxes from initial set
    '''
    boxes_to_filter = tf.pad(tensor=boxes,
                             paddings=[[0, k - tf.shape(boxes)[0]], [0, 0]],
                             mode='CONSTANT',
                             constant_values=0.0) if pad_boxes else boxes
    return tf.gather(params=boxes_to_filter,
                     indices=tf.nn.top_k(boxes_to_filter[:, 1],
                                         k=k,
                                         sorted=sort_boxes).indices,
                     axis=0)
