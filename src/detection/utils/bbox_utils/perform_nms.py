import tensorflow as tf


def perform_nms(single_class, max_output_size, iou_threshold):
    '''
    Perform threshold based Non-maximum suppression on given boxes. Works by selecting box with highest confidence score
    and removing surrounding boxes that exceed the IoU threshold with selected box. This process is repeated with the
    new box excluded, until 'max_output_size' boxes are acquired or no remaining boxes are left.
    :param single_class: Boolean masked tensor containing scores for positive classes
    :param max_output_size: Scalar, number of maximium boxes to keep after NMS
    :param iou_threshold: Intersection over Union threshold for filtering overlapping boxes with current selected
    :return: Tensor of predicted boxes after removing non-maxima
    '''
    scores = single_class[..., 1]

    xmin = tf.expand_dims(single_class[..., -4], axis=-1)
    ymin = tf.expand_dims(single_class[..., -3], axis=-1)
    xmax = tf.expand_dims(single_class[..., -2], axis=-1)
    ymax = tf.expand_dims(single_class[..., -1], axis=-1)

    # tf.image.non_max_suppression expects box format of (ymin, xmin, ymax xmax)
    boxes = tf.concat([ymin, xmin, ymax, xmax], -1)
    maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                  scores=scores,
                                                  max_output_size=max_output_size,
                                                  iou_threshold=iou_threshold,
                                                  name='non_maximum_suppression')
    maxima = tf.gather(params=single_class,
                       indices=maxima_indices,
                       axis=0)

    return maxima
