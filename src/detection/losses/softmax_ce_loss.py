import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy


class SOFTMAX_CE_LOSS:
    def compute(self, y_true, y_pred, ce_type='binary'):
        '''
        Compute cross entropy loss between softmax output predicted bounding boxes and ground truth bounding box
        :param y_true: Ground truth bounding boxes
        :param y_pred: Predicted bounding boxes
        :param ce_type: Type of cross entropy loss; one of {'categorical', 'binary'}
        :return: Cross entropy loss for boxes tensor
        '''
        if ce_type == 'binary':
            cross_entropy_loss = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        elif ce_type == 'categorical':
            cross_entropy_loss = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        else:
            raise Exception(f'Cross-entropy type: "{ce_type}" not found.')

        return cross_entropy_loss(y_true, y_pred).numpy()
