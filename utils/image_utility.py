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


def dilate_image(image, dilation_size=5):
    image = np.array(image).astype(np.float) * 255
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                        (dilation_size, dilation_size))
    dilatation_dst = cv2.dilate(image, element)
    dilatation_dst = dilatation_dst / 255
    dilatation_dst = dilatation_dst.astype(np.int).astype(np.bool)
    return dilatation_dst


def dilate_image2(image, dilatation_size=5):
    image = np.array(image).astype(np.float) * 255
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(image, element)
    return dilatation_dst


def display_image(titles, images):
    if isinstance(images, list):
        for title, image in zip(titles, images):
            cv2.imshow(title, np.array(image))
    else:
        cv2.imshow(titles, np.array(images))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = os.path.join(get_project_path(), "environment/env_test/output/scene/office_world.obj")
    _, obstacle_ids, full_occupancy_map, grid_resolution = pickle.load(open(path, 'rb'))
    full_occupancy_map = np.array(full_occupancy_map).astype(np.float) * 255
    # display_image("occ map", full_occupancy_map)
    dilated_map = dilate_image2(full_occupancy_map, dilatation_size=2)
    display_image(["occ map", "dilated image"], [full_occupancy_map, dilated_map])
    print()
