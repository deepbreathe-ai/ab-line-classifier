import dill
import yaml
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.data.preprocessor import Preprocessor

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def predict_instance(x, model):
    '''
    Runs model prediction on 1 or more input images.
    :param x: Image(s) to predict
    :param model: A Keras model
    :return: A numpy array comprising a list of class probabilities for each prediction
    '''
    y = model.predict(x)  # Run prediction on the images
    return y


def predict_set(model, preprocessing_fn, predict_df):
    '''
    Given a dataset, make predictions for each constituent example.
    :param model: A trained TensorFlow model
    :param preprocessing_fn: Preprocessing function to apply before sending image to model
    :param predict_df: Pandas Dataframe of LUS frames, linking image filenames to labels
    :return: List of predicted classes, array of classwise prediction probabilities
    '''

    # Create dataset and apply preprocessing
    dataset = tf.data.Dataset.from_tensor_slices(
        ([cfg['PATHS']['FRAMES'] + f for f in predict_df['Frame Path'].tolist()], predict_df['Class']))
    preprocessor = Preprocessor(preprocessing_fn)
    dataset = preprocessor.prepare(dataset, shuffle=False, augment=False)

    # Obtain prediction probabilities
    p = model.predict(dataset)

    # Get prediction classes in original labelling system
    test_predictions = predict_df['Class'].to_numpy()
    return test_predictions, p