import tensorflow as tf
from src.detection.utils.ssd_utils import filter_single_class
from src.detection.utils.bbox_utils import top_k


def filter_predictions(batch_item,
                       num_classes,
                       confidence_threshold,
                       nms_max_output_size,
                       iou_threshold,
                       num_predictions):
    '''
    Filter predictions for a given batch item by performing confidence thresholding non-maximum suppression,
    and top-k filtering.
    :param batch_item: Batch containing model output tensors
    :param num_classes: Number of classes
    :param nms_max_output_size: Scalar integer representing max number of boxes retained after NMS
    :param confidence_threshold: Model confidence threshold for designating box to target class
    :param iou_threshold: Intersection over Union threshold for NMS
    :param num_predictions: Scalar integer representing number of boxes (preds) to keep after top K filtering
    '''
    filtered_single_classes = tf.nest.map_structure(tf.stop_gradient,
                                                    tf.map_fn(fn=lambda i: filter_single_class(batch_item,
                                                                                               i,
                                                                                               confidence_threshold,
                                                                                               nms_max_output_size,
                                                                                               iou_threshold),
                                                              elems=tf.range(1, num_classes),
                                                              parallel_iterations=128,
                                                              swap_memory=False,
                                                              fn_output_signature=tf.TensorSpec((None, 6),
                                                                                                dtype=tf.float32)),
                                                    name='loop_over_classes')

    # Concatenate filtered results for all individual classes to one tensor
    filtered_predictions = tf.reshape(tensor=filtered_single_classes, shape=(-1, 6))

    # Perform top-k filtering for batch item or pad if fewer that top-k boxes left
    top_k_boxes = tf.cond(tf.greater_equal(tf.shape(filtered_predictions)[0], num_predictions),
                          top_k(boxes=filtered_predictions, k=num_predictions, pad_boxes=False),
                          top_k(boxes=filtered_predictions, k=num_predictions, pad_boxes=True))

    return top_k_boxes
