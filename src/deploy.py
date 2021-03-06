import os.path

import cv2
import numpy as np
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_50v2_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
import matplotlib.pyplot as plt
import pandas as pd
import onnx
from onnx_tf.backend import prepare

from tensorflow.keras.models import load_model

def AB_classifier_preprocess(image, preprocessing_fn):
    '''
    Given a masked ultrasound image, execute preprocessing steps specific to the AB classifier. Specifically, the image
    is resized to (128, 128) and zero-centered with respect to the ImageNet dataset. The result is an image that is
    ready for the forward pass of the view classifier.
    :image (np.array): A masked image with shape (1, H, W, 3)
    :return (np.array): Preprocessed image with shape (1, 128, 128, 3)
    '''

    N_CHANNELS = 3
    INPUT_SIZE = (128, 128)

    # Resize image
    resized_image = cv2.resize(image[0], INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.reshape((1, INPUT_SIZE[0], INPUT_SIZE[1], N_CHANNELS))

    # Apply scaling function
    preprocessed_image = preprocessing_fn(resized_image)
    return preprocessed_image


def predict_framewise(model_path, vid_path, preds_path):
    '''
    Computes frame-wise predictions for a given clip loaded directly from storage
    :param model_path: path to model to make predictions with
    :param vid_path: path to clip to make frame-wise predictions for (any video file format supported by OpenCV)
    :param preds_path: path to save the frame-wise predictions to
    :return: np array of the computed frame-wise predictions
    '''
    model_ext = os.path.splitext(model_path)[1]
    vc = cv2.VideoCapture(vid_path)
    if model_ext == '.onnx':
        model = prepare(onnx.load(model_path))
    else:
        model = load_model(model_path, compile=False)
    preds = []
    while (True):
        ret, frame = vc.read()
        if not ret:
            break
        frame = np.expand_dims(frame, axis=0)
        preprocessed_frame = AB_classifier_preprocess(frame, vgg16_preprocess)
        if model_ext == '.onnx':
            pred = model.run(preprocessed_frame).output
        else:
            pred = model(preprocessed_frame)
        preds += [pred]
    preds = np.vstack(preds)
    pred_df = pd.DataFrame({'Frame': np.arange(preds.shape[0]), 'A lines': preds[:,0], 'B lines': preds[:,1]})
    pred_df.to_csv(preds_path, index=False)
    return preds

# model_path = 'results/models/cutoffvgg16_final_cropped.h5'
# model_path = 'results/models/ab_model_not_compiled/AB_classifier.onnx'
# mp4_path = 'C:/Users/Blake/Downloads/AB_test/demo.mp4'
# preds_path = 'C:/Users/Blake/Downloads/AB_test/demo.csv'
# preds = predict_framewise(model_path, vid_path, preds_path)

