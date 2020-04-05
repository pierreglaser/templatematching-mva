import os

import numpy as np
import matplotlib.pyplot as plt

from .utils import get_data_dir


def load_facesdb():
    datadir = get_data_dir() / "facesdb"
    if not os.path.exists(datadir):
        raise ValueError(
            "facesdb data not found. you may need to run ``fetch_facesdb``"
        )
    return read_images(1521), read_eye_annotations(1521)


def read_images(image_nos):
    if isinstance(image_nos, int):
        image_nos = range(image_nos)
    images = []
    for image_no in image_nos:
        image = plt.imread(
            get_data_dir() / "facesdb" / f"BioID_{image_no:04}.pgm"
        )
        images.append(image)
    return np.stack(images)


def _read_single_image_eye_annotations(img_no):
    filename = get_data_dir() / "facesdb" / f"BioID_{img_no:04}.eye"
    with open(filename, "r") as f:
        lines = list(f.readlines())
    eyes_position = lines[1].strip("\n").split("\t")
    eyes_position = list(map(int, eyes_position))

    assert len(eyes_position) == 4
    left_eye_position = tuple(eyes_position[:2])
    right_eye_position = tuple(eyes_position[2:4])
    return left_eye_position, right_eye_position


def read_eye_annotations(image_nos):
    if isinstance(image_nos, int):
        image_nos = range(image_nos)
    return np.stack([_read_single_image_eye_annotations(i) for i in image_nos])
