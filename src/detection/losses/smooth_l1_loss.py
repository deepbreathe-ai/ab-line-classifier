import tensorflow as tf


class SMOOTH_L1_LOSS:

    def compute(self, y_true, y_pred, smoothing_point=0.5):
        '''
        Compute smooth L1 loss between predicted bounding boxes and ground truth bounding box
        :param y_true: Ground truth bounding boxes
        :param y_pred: Predicted bounding boxes
        :param smoothing_point: Boundary condition of square loss (normally 1.0, but defaults to 0.5 for this project)
        :return: Smooth L1 loss for boxes tensor
        '''
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        res = tf.where(tf.less(abs_loss, smoothing_point), square_loss, abs_loss - 0.5)

        return tf.reduce_sum(res, axis=-1)
