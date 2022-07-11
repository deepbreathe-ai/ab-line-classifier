import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy


class SSD_LOSS:
    '''
    Loss function as defined in paper by Zhao et. al (2021)
    https://www.sciencedirect.com/science/article/abs/pii/S0925231221004215
    '''

    def __init__(self, alpha=1.0, min_negative_boxes=0, negative_boxes_ratio=3,
                 smoothing_point=0.5, ce_type='binary'):
        '''
        Initialize SSD loss object.
        :param alpha: Float, weight for importance given to regression terms in total loss
        :param min_negative_boxes: Integer, min negative sample boxes to keep if adjusting for class imbalance
        :param negative_boxes_ratio: Desired ratio of negative to positive sample boxes when computing loss
        :param smoothing_point: Boundary condition of square loss (normally 1.0, but defaults to 0.5 for this project)
        :param ce_type: Type of cross entropy loss; one of {'categorical', 'binary'}
        '''
        self.alpha = alpha
        self.min_negative_boxes = min_negative_boxes
        self.negative_boxes_ratio = negative_boxes_ratio
        self.smoothing_point = smoothing_point
        self.ce_type = ce_type

    def compute_total(self, y_true, y_pred):
        '''
        Compute total loss for SSD.
        :param y_true: Tensor of pseudo ground-truth labels from both CPM and DPM
        :param y_pred: Tensor of predicted labels from both CPM and DPM
        :return: Sum of total L1 regression and classification loss for network
        '''
        # Labels and preds have shape: (batch size, module, boxes, n+1 classes + 4 loc preds + 8 anchor values)
        cpm_bbox_true = y_true[:, 0, :, :]
        cpm_bbox_pred = y_pred[:, 0, :, :]

        dpm_bbox_true = y_true[:, 1, :, :]
        dpm_bbox_pred = y_pred[:, 1, :, :]

        # CPM's classification is "objectness" (i.e., whether there is an object or not) instead of multi-class
        cpm_loss = self.compute(cpm_bbox_true, cpm_bbox_pred, objectness_adjustment=True)
        dpm_loss = self.compute(dpm_bbox_true, dpm_bbox_pred)

        return cpm_loss + dpm_loss

    def compute(self, y_true, y_pred, objectness_adjustment=False):
        '''
        Compute loss for individual module (either CPM or DPM) in SSD.
        :param y_true: Tensor of pseudo ground-truth labels; binary on classification axis for CPM
        :param y_pred: Tensor of predicted labels
        :param objectness_adjustment: Boolean indicating whether to reformat class-portion of labels for Objectness
        :return: Sum of L1 regression and classification loss for module
        '''
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.shape(y_true)[1]

        # Calculate smooth L1 loss and softmax cross entropy loss for all boxes
        # Third dim is concatenation of n+1 class preds (:-12), 4 localization preds (-12:-8), and 8 default box values
        # for x,y,w,h + 4 variances (i.e., -8:)
        bbox_true = y_true[:, :, -12:-8]
        bbox_pred = y_pred[:, :, -12:-8]
        class_true = y_true[:, :, :-12]
        class_pred = y_pred[:, :, :-12]

        if objectness_adjustment:
            positive_class_fmt = tf.ones_like if tf.reduce_sum(class_true[:, :, 1:]) > 0 else tf.zeros_like
            class_true = tf.concat([class_true[:, :, 0], positive_class_fmt(class_true[:, :, 0])], axis=-1)

        regression_loss = SSD_LOSS.compute_smooth_l1_loss(bbox_true, bbox_pred, smoothing_point=self.smoothing_point)
        classification_loss = SSD_LOSS.compute_softmax_ce_loss(class_true, class_pred, ce_type=self.ce_type)

        # Get negatives and positives from regression loss
        negatives = class_true[:, :, 0]
        positives = tf.reduce_max(class_true[:, :, 1:], axis=-1)
        num_positives = tf.cast(tf.reduce_sum(positives), tf.int32)

        # Calculate positive regression and classification loss
        pos_regression_loss = tf.reduce_sum(regression_loss * positives, axis=-1)
        pos_classification_loss = tf.reduce_sum(classification_loss * positives, axis=-1)

        # Calculate negative classification loss
        neg_classification_loss = classification_loss * negatives
        num_neg_classification_loss = tf.math.count_nonzero(neg_classification_loss, dtype=tf.int32)
        num_neg_classification_loss_keep = tf.minimum(
            tf.maximum(self.negative_boxes_ratio * num_positives, self.min_negative_boxes),
            num_neg_classification_loss)

        # Most matched default boxes are of background class, to help with imbalance perform hard negative mining
        neg_classification_loss = tf.cond(tf.equal(num_neg_classification_loss, tf.constant(0)),
                                          lambda: tf.zeros([batch_size]),
                                          lambda: SSD_LOSS.hard_negative_mining(neg_classification_loss,
                                                                                classification_loss,
                                                                                num_neg_classification_loss_keep,
                                                                                batch_size,
                                                                                num_boxes))
        classification_loss = pos_classification_loss + neg_classification_loss

        total = (classification_loss + self.alpha * pos_regression_loss) / tf.maximum(1.0, tf.cast(num_positives,
                                                                                                   tf.float32))
        return total

    @staticmethod
    def compute_smooth_l1_loss(y_true, y_pred, smoothing_point=0.5):
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

    @staticmethod
    def compute_softmax_ce_loss(y_true, y_pred, ce_type='binary'):
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

    @staticmethod
    def hard_negative_mining(negative_classification_loss, classification_loss,
                             num_neg_to_keep, batch_size, num_boxes):
        '''
        Performs hard negative mining by selecting the top negative boxes based on confidence and
        recalculates the negative classification loss using only the selected top negatives.
        :param negative_classification_loss: Initial negative sample classification loss prior to mining
        :param classification_loss: Total classification loss
        :param num_neg_to_keep: Number of negative samples to keep in new loss calculation
        :param batch_size: Batch size
        :param num_boxes: Total number of anchor boxes
        :return: Updated negative classification loss value based on top negative samples
        '''
        negative_classification_loss_1d = tf.reshape(negative_classification_loss, [-1])
        _, indices = tf.nn.top_k(negative_classification_loss_1d, k=num_neg_to_keep, sorted=False)
        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(negative_classification_loss_1d))
        negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, num_boxes]), tf.float32)
        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)

        return neg_class_loss
