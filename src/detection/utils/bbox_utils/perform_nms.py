import tensorflow as tf


def perform_nms(single_class, max_output_size, iou_threshold):
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
