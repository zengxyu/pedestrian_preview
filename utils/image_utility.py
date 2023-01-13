import os
import pickle

import cv2
import numpy as np

from utils.fo_utility import get_project_path


def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE


def dilate_image(image, dilation_size):
    image = np.array(image).astype(np.float) * 255
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                        (dilation_size, dilation_size))
    dilatation_dst = cv2.dilate(image, element)
    dilatation_dst = dilatation_dst / 255
    dilatation_dst = dilatation_dst.astype(np.int).astype(np.bool)
    return dilatation_dst
