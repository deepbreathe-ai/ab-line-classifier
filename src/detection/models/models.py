import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, GlobalAveragePooling2D, Conv2D, MaxPool2D, \
    Reshape, Concatenate, BatchNormalization, AveragePooling2D, Flatten, SpatialDropout2D, Conv2DTranspose
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
import yaml
import os
from src.detection.custom_layers import DefaultBoxes, DecodeSSDPredictions
from src.detection.utils.ssd_utils import get_default_box_count

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))


def get_model(model_name):
    '''
    Return the model definition and associated preprocessing function as specified in the config file
    :return: (TF model definition function, preprocessing function)
    '''

    if model_name == 'vgg16':
        model_def = vgg16
        preprocessing_function = vgg16_preprocess
    elif model_name == 'cutoffvgg16':
        model_def = CutoffVGG16
        preprocessing_function = vgg16_preprocess
    else:
        model_def = SSD_CutoffVGG16
        preprocessing_function = vgg16_preprocess
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
        self.cutoff_layer = model_config['CUTOFF_LAYER']
        self.finetune_layer = model_config['FINETUNE_LAYER']
        self.extract_epochs = model_config['EXTRACT_EPOCHS']
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

        # dpm_transpose2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same',
        #                                  strides=(2, 2),
        #                                  kernel_initializer=tf.constant_initializer(cpm2_conv3.kernel[...].numpy()),
        #                                  bias_initializer=tf.constant_initializer(cpm2_conv3.bias[...].numpy()))(
        #     dpm_transpose1)
        dpm_transpose2 = Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same',
                                         strides=(2, 2), name='dpm_transpose2')(dpm_transpose1)
        dpm_transpose2 = dpm_transpose2 * cpm2_conv3.output

        # dpm_transpose3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
        #                                   strides=(2, 2),
        #                                   kernel_initializer=tf.constant_initializer(cpm_1_conv_3.kernel[...].numpy()),
        #                                   bias_initializer=tf.constant_initializer(cpm_1_conv_3.bias[...].numpy()))(dpm_transpose2)
        dpm_transpose3 = Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
                                         strides=(2, 2), name='dpm_transpose3')(dpm_transpose2)
        dpm_transpose3 = dpm_transpose3 * cpm1_conv2.output

        ssd_model = Model(inputs=self.base_model.input, outputs=dpm_transpose3)

        # Construct prediction layers (conf, loc, and default boxes)
        scales = np.linspace(self.min_scale, self.max_scale, len(self.prediction_layers))
        num_cls_with_bg = self.n_classes + 1

        mbox_conf_layers = []
        mbox_loc_layers = []
        mbox_default_box_layers = []

        for i, layer in enumerate(self.prediction_layers):
            default_box_count = get_default_box_count(layer['aspect_ratios'],
                                                      self.extra_box_for_ar1)
            x = ssd_model.get_layer(layer['name']).output
            layer_mbox_conf = Conv2D(filters=default_box_count * num_cls_with_bg,
                                     kernel_size=(3, 3),
                                     padding='same',  # Potentially add kernel init and regularizers
                                     name=f"{layer['name']}_mbox_conf")(x)
            layer_mbox_conf_reshape = Reshape((-1, num_cls_with_bg),
                                              name=f"{layer['name']}_mbox_conf_reshape")(layer_mbox_conf)

            layer_mbox_loc = Conv2D(filters=default_box_count * 4,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    name=f"{layer['name']}_mbox_loc")(x)
            layer_mbox_loc_reshape = Reshape((-1, 4),
                                             name=f"{layer['name']}_mbox_loc_reshape")(layer_mbox_loc)

            layer_default_boxes = DefaultBoxes(image_shape=self.input_shape,
                                               scale=scales[i],
                                               next_scale=scales[i + 1] if i + 1 <= len(
                                                   self.prediction_layers) - 1 else 1,
                                               aspect_ratios=layer['aspect_ratios'],
                                               variances=self.variances,
                                               has_extra_box_for_ar_1=self.extra_box_for_ar1,
                                               name=f"{layer['name']}_default_boxes")(x)
            layer_default_boxes_reshape = Reshape((-1, 8),
                                                  name=f"{layer['name']}_default_boxes_reshape")(layer_default_boxes)

            mbox_conf_layers.append(layer_mbox_conf_reshape)
            mbox_loc_layers.append(layer_mbox_loc_reshape)
            mbox_default_box_layers.append(layer_default_boxes_reshape)

        # Concatenate class confidence predictions from different feature map layers
        mbox_conf = Concatenate(axis=-2, name='mbox_conf')(mbox_conf_layers)
        mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

        # Concatenate object location predictions from different feature map layers
        mbox_loc = Concatenate(axis=-2, name='mbox_loc')(mbox_loc_layers)

        # Concatenate default boxes from different feature map layers
        mbox_default_boxes = Concatenate(axis=-2, name='mbox_default_boxes')(mbox_default_box_layers)

        # Concatenate confidence score predictions, bounding box predictions, and default boxes
        predictions = Concatenate(axis=-1, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_default_boxes])

        ssd_pred_model = Model(inputs=self.base_model.input, outputs=predictions)
        ssd_pred_model.summary()

        return ssd_pred_model

    def fit(self, train_data, steps_per_epoch=None, epochs=1, validation_data=None, validation_steps=None,
            callbacks=None, verbose=1, class_weight=None):
        for layer in self.vgg16_layers:
            layer.trainable = False
        self.model.compile(optimizer=self.optimizer_extract, loss='categorical_crossentropy', metrics=self.metrics,
                           run_eagerly=True)
        history_extract = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=self.extract_epochs,
                                         validation_data=validation_data, validation_steps=validation_steps,
                                         callbacks=callbacks,
                                         verbose=verbose, class_weight=class_weight)
        for layer in self.vgg16_layers[self.finetune_layer:]:
            layer.trainable = True
        self.model.compile(optimizer=self.optimizer_finetune, loss='categorical_crossentropy', metrics=self.metrics,
                           run_eagerly=True)
        history_finetune = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                          initial_epoch=history_extract.epoch[-1],
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


def vgg16(model_config, input_shape, metrics, n_classes, mixed_precision, output_bias=None):
    '''
    Defines a model based on a pretrained VGG16 for binary US classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param mixed_precision: Whether to use mixed precision (use if you have GPU with compute capacity >= 7.0)
    :param output_bias: bias initializer of output layer
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    weight_decay = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    frozen_layers = model_config['FROZEN_LAYERS']
    fc0_nodes = model_config['NODES_DENSE0']

    print("MODEL CONFIG: ", model_config)

    if mixed_precision:
        tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    if output_bias is not None:
        output_bias = Constant(output_bias)  # Set initial output bias

    # Start with pretrained VGG16
    X_input = Input(input_shape, name='input')
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)

    # Freeze layers
    for layers in range(len(frozen_layers)):
        layer2freeze = frozen_layers[layers]
        print('Freezing layer: ' + str(layer2freeze))
        base_model.layers[layer2freeze].trainable = False

    X = base_model.output

    # Add custom top layers
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(n_classes, bias_initializer=output_bias, name='logits')(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


class CutoffVGG16:

    def __init__(self, model_config, input_shape, metrics, n_classes, mixed_precision=False, output_bias=None):
        self.lr_extract = model_config['LR_EXTRACT']
        self.lr_finetune = model_config['LR_FINETUNE']
        self.dropout = model_config['DROPOUT']
        self.cutoff_layer = model_config['CUTOFF_LAYER']
        self.finetune_layer = model_config['FINETUNE_LAYER']
        self.extract_epochs = model_config['EXTRACT_EPOCHS']
        self.optimizer_extract = Adam(learning_rate=self.lr_extract)
        self.optimizer_finetune = RMSprop(learning_rate=self.lr_finetune)
        self.output_bias = output_bias
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.metrics = metrics
        self.mixed_precision = mixed_precision
        self.model = self.define_model()

    def define_model(self):
        X_input = Input(shape=self.input_shape, name='input')
        vgg16 = VGG16(input_shape=self.input_shape, include_top=False, weights='imagenet')
        self.vgg16_layers = vgg16.layers[1:self.cutoff_layer]
        X = X_input
        for layer in self.vgg16_layers:
            X = layer(X)
        X = GlobalAveragePooling2D(name='global_avgpool')(X)
        X = Dropout(self.dropout)(X)
        Y = Dense(self.n_classes, activation='softmax', bias_initializer=self.output_bias, name='output')(X)
        model = Model(inputs=X_input, outputs=Y)
        model.summary()
        return model

    def fit(self, train_data, steps_per_epoch=None, epochs=1, validation_data=None, validation_steps=None,
            callbacks=None, verbose=1, class_weight=None):
        for layer in self.vgg16_layers:
            layer.trainable = False
        self.model.compile(optimizer=self.optimizer_extract, loss='categorical_crossentropy', metrics=self.metrics,
                           run_eagerly=True)
        history_extract = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=self.extract_epochs,
                                         validation_data=validation_data, validation_steps=validation_steps,
                                         callbacks=callbacks,
                                         verbose=verbose, class_weight=class_weight)
        for layer in self.vgg16_layers[self.finetune_layer:]:
            layer.trainable = True
        self.model.compile(optimizer=self.optimizer_finetune, loss='categorical_crossentropy', metrics=self.metrics,
                           run_eagerly=True)
        history_finetune = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                          initial_epoch=history_extract.epoch[-1],
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
