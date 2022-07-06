import tensorflow as tf
from src.detection.utils.ssd_utils import filter_predictions


def decode_predictions(
        y_pred,
        input_size,
        nms_max_output_size=400,
        confidence_threshold=0.01,
        iou_threshold=0.45,
        num_predictions=10
):
    '''
    Decode bounding box predictions
    #TODO: Fill in documentation for this function and its parameters
    '''

    cx = y_pred[..., -12] * y_pred[..., -4] * y_pred[..., -6] + y_pred[..., -8]
    cy = y_pred[..., -11] * y_pred[..., -3] * y_pred[..., -5] + y_pred[..., -7]
    width = tf.exp(y_pred[..., -10] * tf.sqrt(y_pred[..., 2])) * y_pred[..., -6]
    height = tf.exp(y_pred[..., -9] * tf.sqrt(y_pred[..., -1])) * y_pred[..., -5]

    # Convert bboxes to corner format & scale by input size
    xmin = (cx - 0.5 * width) * input_size
    ymin = (cy - 0.5 * height) * input_size
    xmax = (cx + 0.5 * width) * input_size
    ymax = (cy + 0.5 * height) * input_size

    # Concatenate class and bbox predictions
    y_pred = tf.concat([y_pred[..., :-12],
                        tf.expand_dims(xmin, axis=-1),
                        tf.expand_dims(ymin, axis=-1),
                        tf.expand_dims(xmax, axis=-1),
                        tf.expand_dims(ymax, axis=-1)], -1)

    batch_size = tf.shape(y_pred)[0]  # dtype: tf.int32
    num_boxes = tf.shape(y_pred)[1]
    num_classes = y_pred.shape[2] - 4
    class_indices = tf.range(1, num_classes)

    # Iterate filter_predictions() over all batch items
    output_tensor = tf.nest.map_structure(tf.stop_gradient,
                                          tf.map_fn(fn=lambda x: filter_predictions(x,
                                                                                    num_classes,
                                                                                    confidence_threshold,
                                                                                    nms_max_output_size,
                                                                                    iou_threshold,
                                                                                    num_predictions),
                                                    elems=y_pred,
                                                    parallel_iterations=128,
                                                    swap_memory=False,
                                                    fn_output_signature=tf.TensorSpec((num_predictions, 6),
                                                                                      dtype=tf.float32),
                                                    name='loop_over_batch'))

    return output_tensor
