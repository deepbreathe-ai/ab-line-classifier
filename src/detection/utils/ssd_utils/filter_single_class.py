import tensorflow as tf
from src.detection.utils.bbox_utils import perform_nms


def filter_single_class(batch_item,
                        index,
                        confidence_threshold,
                        nms_max_output_size,
                        iou_threshold):
    '''
    Filter predictions from SSD for one single class
    :param batch_item: Batch containing model output tensors
    :param index: Index of class
    :param nms_max_output_size: Scalar integer representing max number of boxes retained after NMS
    :param confidence_threshold: Model confidence threshold for designating box to target class
    :param iou_threshold: Intersection over Union threshold for NMS
    '''
    confidences = tf.expand_dims(batch_item[..., index], axis=-1)
    class_id = tf.fill(dims=tf.shape(confidences),
                       value=tf.cast(index, tf.float32))
    box_coordinates = batch_item[..., -4:]

    single_class = tf.concat([class_id, confidences, box_coordinates], -1)

    # Apply confidence thresholding w.r.t class defined by 'index'
    threshold_met = single_class[:, 1] > confidence_threshold
    single_class = tf.boolean_mask(tensor=single_class, mask=threshold_met)

    single_class_nms = tf.cond(
        tf.equal(tf.size(single_class), 0),
        lambda: tf.constant(value=0.0, shape=(1, 6)),
        perform_nms(single_class=single_class,
                    max_output_size=nms_max_output_size,
                    iou_threshold=iou_threshold))

    # Ensure single_class is exactly nms_max_output_size elements long
    padded_single_class = tf.pad(tensor=single_class_nms,
                                 paddings=[[0, nms_max_output_size - tf.shape(single_class_nms)[0]], [0, 0]],
                                 mode='CONSTANT',
                                 constant_values=0.0)

    return padded_single_class
