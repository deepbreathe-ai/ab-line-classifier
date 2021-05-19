import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import *
from tensorflow.keras import layers
import os, yaml

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

class Preprocessor:
    '''
    Responsible for creating a TF dataset from frame filenames and corresponding labels. Includes data preprocessing
    functionality, including resizing and data augmentation.
    '''

    def __init__(self, preprocess_fn):
        '''
        @param preprocess_fn: Model-specific preprocessing function
        '''
        self.batch_size = cfg['TRAIN']['BATCH_SIZE']
        self.n_classes = len(cfg['DATA']['CLASSES'])
        self.img_dir = cfg['PATHS']['FRAMES']
        self.autotune = tf.data.AUTOTUNE
        self.data_augmentation = tf.keras.Sequential([
            RandomBrightness(factor=cfg['TRAIN']['DATA_AUG']['BRIGHTNESS_RANGE']),
            RandomContrast(cfg['TRAIN']['DATA_AUG']['CONTRAST_RANGE']),
            RandomFlip("horizontal"),
            RandomRotation(cfg['TRAIN']['DATA_AUG']['ROTATION_RANGE'], fill_mode='nearest'),
            RandomZoom(cfg['TRAIN']['DATA_AUG']['ZOOM_RANGE'], fill_mode='nearest')
        ])
        self.input_scaler = preprocess_fn

    def prepare(self, ds, shuffle=False, augment=False):
        '''
        Maps a series of preprocessing functions to each item in a dataset
        @param ds: A TF Dataset
        @param shuffle: Flag indicating whether to shuffle the dataset
        @type augment: Flag indicating whether to include data augmentation transformations
        @return: The prepared TF dataset
        '''

        # Load and resize images
        ds = ds.map(self._parse_fn, num_parallel_calls=self.autotune)

        # Shuffle items in the dataset
        if shuffle:
            ds = ds.shuffle(1000)

        # Batch the dataset
        ds = ds.batch(self.batch_size)

        # Conduct data augmentation transformations, if indicated
        if augment:
            ds = ds.map(lambda x, y: (self.data_augmentation(x, training=True), y), num_parallel_calls=self.autotune)

        # Apply input preprocessing function to scale the inputs
        ds = ds.map(lambda x, y: (self.input_scaler(x), y), num_parallel_calls=self.autotune)  # Scale inputs

        # Apply buffered prefetching on dataset
        return ds.prefetch(buffer_size=self.autotune)


    def _parse_fn(self, filename, label):
        '''
        Parse image file and resize image. Produces a tuple consisting of a resized image and its corresponding label.
        @param filename: File name of the image
        @param label: Label assigned to the image
        @return: (image, label) tuple consisting of the  resized image and its class
        '''
        image_str = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_str, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        return tf.image.resize(image, cfg['DATA']['IMG_DIM']), tf.one_hot(label, self.n_classes)



class RandomBrightness(layers.Layer):
    '''
    A custom layer that applies a ranom brightness shift to an image
    '''

    def __init__(self, factor=0.5, **kwargs):
        '''
        @param factor: Absolute value of the maximum brightness shift, in [0, 1]
        '''
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, image):
        '''
        Applies random brightness shift to input image
        @param image: image tensor
        @return: image tensor with random amount of brightness applied to it
        '''
        return tf.image.stateless_random_brightness(image, self.factor, (123, 0))
