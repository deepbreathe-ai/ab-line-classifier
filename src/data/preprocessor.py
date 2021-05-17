import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import *
import os, yaml

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

class Preprocessor:

    def __init__(self, preprocess_fn):
        self.batch_size = cfg['TRAIN']['BATCH_SIZE']
        self.n_classes = len(cfg['DATA']['CLASSES'])
        self.img_dir = cfg['PATHS']['FRAMES']
        self.autotune = tf.data.AUTOTUNE
        self.data_augmentation = tf.keras.Sequential([
            RandomFlip("horizontal"),
            RandomRotation(cfg['TRAIN']['DATA_AUG']['ROTATION_RANGE'], fill_mode='constant'),
            RandomZoom(cfg['TRAIN']['DATA_AUG']['ZOOM_RANGE'])
        ])
        self.input_scaler = preprocess_fn

    def prepare(self, ds, shuffle=False, augment=False):

        ds = ds.map(self._parse_fn, num_parallel_calls=self.autotune)

        # ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(1000)

        # Batch all datasets
        ds = ds.batch(self.batch_size)

        # Use data augmentation only on the training set
        if augment:
            ds = ds.map(lambda x, y: (self.data_augmentation(x, training=True), y), num_parallel_calls=self.autotune)

        ds = ds.map(lambda x, y: (self.input_scaler(x), y), num_parallel_calls=self.autotune)  # Scale inputs

        # Use buffered prefecting on all datasets
        return ds.prefetch(buffer_size=self.autotune)


    def _parse_fn(self, filename, label):
        image_str = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_str, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        return tf.image.resize(image, cfg['DATA']['IMG_DIM']), tf.one_hot(label, self.n_classes)
