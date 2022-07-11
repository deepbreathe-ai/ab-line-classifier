import cv2
import numpy as np


def extract_bbox_from_heatmap(heatmap, threshold=None, max_val=255, thresh_type=cv2.THRESH_BINARY, is_grayscale=True,
                              perform_otsu=False, perform_dilation=False, dilation_kernel_size=(1, 5), dilation_iters=1,
                              min_contour_size=20):
    '''
    Extract bounding box coordinated from a heatmap (saliency map) around the highest activations.
    :param heatmap: Original heatmap
    :param threshold: Integer value for thresholding cutoff
    :param max_val: Integer value for max pixel value
    :param thresh_type: Thresholding technique, see enums in opencv-python
    :param is_grayscale: Boolean indicating whether input heatmap is already in 3-dim grayscale; assumed RGB otherwise
    :param perform_otsu: Boolean indicating whether to use Otsu's Thresholding instead of defined threshold value
    :param perform_dilation: Boolean indicating whether to perform dilation on binarized mask
    :param dilation_kernel_size: Tuple of kernel shape used for dilation if 'dilate' is set to True
    :param dilation_iters: Number of times to run dilation if 'dilate' is set to True
    :param min_contour_size: Integer for min number for points in retained contours; used for filtering weak contours
    :return: Bounding box coordinates around highest activation in heatmap
    '''
    # Set threshold to half of max val if none provided
    if threshold is None:
        threshold = max_val // 2

    gray_heatmap = heatmap[:, :, 0] if is_grayscale else cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    ret, thresh = cv2.threshold(gray_heatmap,
                                thresh=threshold,
                                maxval=max_val,
                                type=thresh_type + (int(perform_otsu)*cv2.THRESH_OTSU))

    if perform_dilation:
        thresh = cv2.dilate(thresh, np.ones(dilation_kernel_size, np.uint8), iterations=dilation_iters)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    all_boxes = []
    for contour in contours:
        if len(contour) > min_contour_size:
            x, y, w, h = cv2.boundingRect(contour)

            all_boxes.append([x, y, w, h])

    # TODO: Identify and filter out "poor" boxes further
    # TODO: Experiment with adaptively setting threshold value based on model confidence in GT class

    return all_boxes
