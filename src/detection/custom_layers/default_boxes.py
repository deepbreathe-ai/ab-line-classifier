import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from src.detection.utils.ssd_utils import generate_default_boxes_for_feature_map


class DefaultBoxes(Layer):
    '''
    Custom TF layer to help generate default boxes for a given feature map.
    '''
    # TODO: Document method parameters

    def __init__(self,
                 image_shape,
                 scale,
                 next_scale,
                 aspect_ratios,
                 variances,
                 has_extra_box_for_ar_1=True,
                 clip_boxes=True,
                 offset=(0.5, 0.5),
                 **kwargs):
        self.image_shape = image_shape
        self.scale = scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.has_extra_box_for_ar_1 = has_extra_box_for_ar_1
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.offset = offset
        super(DefaultBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        _, feature_map_height, feature_map_width, _ = input_shape
        image_height, image_width, _ = self.image_shape

        assert(feature_map_height == feature_map_width, "DefaultBoxes requires a square feature map")
        assert(image_height == image_width, "DefaultBoxes requires a square image")

        self.feature_map_size = feature_map_height
        self.image_size = image_height
        super(DefaultBoxes, self).build(input_shape)

    def call(self, inputs):
        default_boxes = generate_default_boxes_for_feature_map(
            feature_map_size=self.feature_map_size,
            image_size=self.image_size,
            offset=self.offset,
            scale=self.scale,
            next_scale=self.next_scale,
            aspect_ratios=self.aspect_ratios,
            variances=self.variances,
            has_extra_box_for_ar_1=self.has_extra_box_for_ar_1,
            clip_boxes=self.clip_boxes
        )
        default_boxes = np.expand_dims(default_boxes, axis=0)
        default_boxes = tf.constant(default_boxes, dtype='float32')
        default_boxes = tf.tile(default_boxes, (tf.shape(inputs)[0], 1, 1, 1, 1))

        return default_boxes

    def get_config(self):
        config = {
            "image_shape": self.image_shape,
            "scale": self.scale,
            "next_scale": self.next_scale,
            "aspect_ratios": self.aspect_ratios,
            "has_extra_box_for_ar_1": self.has_extra_box_for_ar_1,
            "clip_boxes": self.clip_boxes,
            "variances": self.variances,
            "offset": self.offset,
            "feature_map_size": self.feature_map_size,
            "image_size": self.image_size
        }

        base_config = super(DefaultBoxes. self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
