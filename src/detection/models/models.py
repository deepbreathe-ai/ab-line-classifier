import os
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.layers import Activation, Conv2D, MaxPool2D, \
    Reshape, Concatenate, Conv2DTranspose
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from src.detection.custom_layers import DefaultBoxes
from src.detection.utils.ssd_utils import get_default_box_count

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))


def get_model(model_name):
    '''
    Return the model definition and associated preprocessing function as specified in the config file
    :return: (TF model definition function, preprocessing function)
    '''

    if model_name == 'ssd_cutoffvgg16':
        model_def = SSD_CutoffVGG16
        preprocessing_function = vgg16_preprocess
    else:
        raise Exception(f'Model definition: "{model_name}" not found')

    return model_def, preprocessing_function


class SSD_CutoffVGG16:
    '''
    Defines a model based on Zhao et al. (2018)'s SSD architecture with a Cutoff VGG16 backbone.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    def __init__(self, model_config, backbone_path, input_shape, metrics, n_classes, mixed_precision=False,
                 output_bias=None):
        self.lr_extract = model_config['LR_EXTRACT']
        self.lr_finetune = model_config['LR_FINETUNE']
        self.dropout = model_config['DROPOUT']
        self.epochs = model_config['EPOCHS']
        self.frozen_layers = model_config['FROZEN_LAYERS']
        self.optimizer_extract = Adam(learning_rate=self.lr_extract)
        self.optimizer_finetune = RMSprop(learning_rate=self.lr_finetune)
        self.output_bias = output_bias
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.metrics = metrics
        self.mixed_precision = mixed_precision
        self.backbone_path = backbone_path
        self.base_model = None
        self.prediction_layers = model_config['PREDICTION_LAYERS']
        self.extra_box_for_ar1 = model_config['EXTRA_BOX_FOR_AR1']
        self.min_scale = model_config['MIN_SCALE']
        self.max_scale = model_config['MAX_SCALE']
        self.variances = model_config['VARIANCES']
        self.model = self.define_model()

    def define_model(self):
        # Load pretrained backbone
        self.base_model = load_model(self.backbone_path, compile=False)
        self.base_model.summary()
        # Rename shared layers to match with SSD module, block nums decremented when going from base to SSD
        for layer in self.base_model.layers:
            if 'block2' in layer.name:
                layer._name = str.replace(layer.name, 'block2', 'cpm1')
            elif 'block3' in layer.name:
                layer._name = str.replace(layer.name, 'block3', 'cpm2')

        # Build the Single-Shot Detector Module (SDM)
        # Convolution Prediction Module (CPM) - first two conv blocks are from backbone
        cpm1_conv2 = self.base_model.get_layer('cpm1_conv2')

        # Add max pooling when branching from base network to SSD
        cpm2_conv3 = self.base_model.get_layer('cpm2_conv3')
        cpm2_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='cpm2_pool')(cpm2_conv3.output)

        cpm3_conv1 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='cpm3_conv1')(cpm2_pool)
        cpm3_conv2 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='cpm3_conv2')(cpm3_conv1)
        cpm3_conv3 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', name='cpm3_conv3')(cpm3_conv2)
        cpm3_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='cpm3_pool')(cpm3_conv3)

        # Deconvolution Prediction Module (DPM)
        # TODO: Potentially replace 'Conv2DTranspose' with max unpooling + conv
        dpm_transpose1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), padding='same',
                                         strides=(2, 2), name='dpm_transpose1')(cpm3_pool)
        dpm_transpose1 = dpm_transpose1 * cpm3_conv3  # Fuse with original feature map

        # The last two DPM layers can initialize off reshaped kernel weights of first two CPM blocks which come from
        # the trained backbone.
        cpm2_conv3_kernel = tf.constant_initializer(np.tile(cpm2_conv3.kernel[...].numpy(), (1, 1, 1, 2)))
        dpm_transpose2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same',
                                         strides=(2, 2),
                                         kernel_initializer=cpm2_conv3_kernel,
                                         bias_initializer=tf.constant_initializer(cpm2_conv3.bias[...].numpy()),
                                         name='dpm_transpose2')(dpm_transpose1)
        dpm_transpose2 = dpm_transpose2 * cpm2_conv3.output

        cpm1_conv2_kernel = tf.constant_initializer(np.tile(cpm1_conv2.kernel[...].numpy(), (1, 1, 1, 2)))
        dpm_transpose3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
                                         strides=(2, 2),
                                         kernel_initializer=cpm1_conv2_kernel,
                                         bias_initializer=tf.constant_initializer(cpm1_conv2.bias[...].numpy()),
                                         name='dpm_transpose3')(dpm_transpose2)
        dpm_transpose3 = dpm_transpose3 * cpm1_conv2.output

        ssd_model = Model(inputs=self.base_model.input, outputs=dpm_transpose3)

        # Construct prediction layers (conf, loc, and default boxes)
        scales = [np.linspace(self.min_scale, self.max_scale, len(self.prediction_layers[0]['CPM'])),
                  np.linspace(self.max_scale, self.min_scale, len(self.prediction_layers[1]['DPM']))]
        num_cls_with_bg = self.n_classes + 1

        mbox_conf_layers = [[], []]
        mbox_loc_layers = [[], []]
        mbox_default_box_layers = [[], []]

        for module in self.prediction_layers:
            module_name = next(iter(module))
            pred_set_idx = 0 if module_name == 'CPM' else 1  # Need to segregate since CPM and DPM have different losses
            for i, layer in enumerate(module[module_name]):
                default_box_count = get_default_box_count(layer['aspect_ratios'],
                                                          self.extra_box_for_ar1)
                x = ssd_model.get_layer(layer['name']).output
                layer_mbox_conf = Conv2D(filters=default_box_count * num_cls_with_bg,
                                         kernel_size=(3, 3),
                                         padding='same',  # TODO: Potentially add kernel init and L2 regularizers
                                         name=f"{layer['name']}_mbox_conf")(x)
                layer_mbox_conf_reshape = Reshape((-1, num_cls_with_bg),
                                                  name=f"{layer['name']}_mbox_conf_reshape")(layer_mbox_conf)

                layer_mbox_loc = Conv2D(filters=default_box_count * 4,
                                        kernel_size=(3, 3),
                                        padding='same',
                                        name=f"{layer['name']}_mbox_loc")(x)
                layer_mbox_loc_reshape = Reshape((-1, 4),
                                                 name=f"{layer['name']}_mbox_loc_reshape")(layer_mbox_loc)

                # TODO: Forward refined default boxes from CPM to DPM
                layer_default_boxes = DefaultBoxes(image_shape=self.input_shape,
                                                   scale=scales[pred_set_idx][i],
                                                   next_scale=scales[pred_set_idx][i + 1] if i + 1 <= len(
                                                       module[module_name]) - 1 else 1,
                                                   aspect_ratios=layer['aspect_ratios'],
                                                   variances=self.variances,
                                                   has_extra_box_for_ar_1=self.extra_box_for_ar1,
                                                   name=f"{layer['name']}_default_boxes")(x)
                layer_default_boxes_reshape = Reshape((-1, 8),
                                                      name=f"{layer['name']}_default_boxes_reshape")(layer_default_boxes)

                mbox_conf_layers[pred_set_idx].append(layer_mbox_conf_reshape)
                mbox_loc_layers[pred_set_idx].append(layer_mbox_loc_reshape)
                mbox_default_box_layers[pred_set_idx].append(layer_default_boxes_reshape)

        # Concatenate class confidence predictions from different feature map layers
        mbox_conf = [Concatenate(axis=-2, name=f'{i}_mbox_conf')(mbox_conf_layers[i])
                     for i in range(len(mbox_conf_layers))]
        mbox_conf_softmax = [Activation('softmax', name=f'{i}_mbox_conf_softmax')(mbox_conf[i])
                             for i in range(len(mbox_conf))]
        mbox_conf_softmax = tf.stack(mbox_conf_softmax, axis=1)

        # Concatenate object location predictions from different feature map layers
        mbox_loc = [Concatenate(axis=-2, name=f'{i}_mbox_loc')(mbox_loc_layers[i])
                    for i in range(len(mbox_loc_layers))]
        mbox_loc = tf.stack(mbox_loc, axis=1)

        # Concatenate default boxes from different feature map layers
        mbox_default_boxes = [Concatenate(axis=-2, name=f'{i}_mbox_default_boxes')(mbox_default_box_layers[i])
                              for i in range(len(mbox_default_box_layers))]
        mbox_default_boxes = tf.stack(mbox_default_boxes, axis=1)

        # Concatenate confidence score predictions, bounding box predictions, and default boxes
        predictions = [Concatenate(axis=-1, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_default_boxes])]

        ssd_pred_model = Model(inputs=self.base_model.input, outputs=predictions)
        ssd_pred_model.summary()

        return ssd_pred_model

    def fit(self, train_data, steps_per_epoch=None, epochs=None, validation_data=None, validation_steps=None,
            callbacks=None, verbose=1, class_weight=None):
        epochs = self.epochs if epochs is None else epochs  # Function caller can override

        # Freeze specified layers
        for idx in self.frozen_layers:
            layer = self.model.get_layer(index=idx)
            layer.trainable = False

        # TODO: Finish implementation of this function (fitting to custom data generator)

        self.model.compile(optimizer=self.optimizer_extract, loss='categorical_crossentropy', metrics=self.metrics,
                           run_eagerly=True)
        history_extract = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                         validation_data=validation_data, validation_steps=validation_steps,
                                         callbacks=callbacks,
                                         verbose=verbose, class_weight=class_weight)

    def evaluate(self, test_data, verbose=1):
        return self.model.evaluate(test_data, verbose=verbose)

    def predict(self, test_data, verbose=1):
        return self.model.predict(test_data, verbose=verbose)

    @property
    def metrics_names(self):
        return self.model.metrics_names


if __name__ == '__main__':
    hparams = cfg['HPARAMS'][cfg['TRAIN']['MODEL_DEF'].upper()]
    backbone_model_path = cfg['PATHS']['BACKBONE_MODEL']
    ssd = SSD_CutoffVGG16(hparams, backbone_model_path, (128, 128, 3), None, 2)
    ssd.define_model()
