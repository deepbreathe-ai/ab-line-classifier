from tensorflow.keras.layers import Layer
from src.detection.utils.ssd_utils import decode_predictions


class DecodeSSDPredictions(Layer):
    '''
    Layer used for decoding predictions when running trained SSD model in inference mode.
    '''
    def __init__(self,
                 input_size,
                 nms_max_output_size=400,
                 confidence_threshold=0.01,
                 iou_threshold=0.45,
                 num_predictions=10,
                 **kwargs):
        '''
        Initialize decoding layer.
        :param input_size: Tuple, input size of image
        :param nms_max_output_size: Scalar integer representing max number of boxes retained after NMS
        :param confidence_threshold: Model confidence threshold for designating box to target class
        :param iou_threshold: Intersection over Union threshold for NMS
        :param num_predictions: Scalar integer representing number of boxes (preds) to keep after top K filtering
        '''
        self.input_size = input_size
        self.nms_max_output_size = nms_max_output_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_predictions = num_predictions

        super(DecodeSSDPredictions, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DecodeSSDPredictions, self).build(input_shape)

    def call(self, inputs):
        y_pred = decode_predictions(
            y_pred=inputs,
            input_size=self.input_size,
            nms_max_output_size=self.nms_max_output_size,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            num_predictions=self.num_predictions
        )

        return y_pred

    def get_config(self):
        config = {
            'input_size': self.input_size,
            'nms_max_output_size': self.nms_max_output_size,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'num_predictions': self.num_predictions,
        }

        base_config = super(DecodeSSDPredictions, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
