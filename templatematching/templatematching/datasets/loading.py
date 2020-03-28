import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "images"
PROCESS_DATA_DIR = Path(__file__).parent.parent.parent / "img_normalized"
PATCH_DIR = Path(__file__).parent.parent.parent / "patches"


def read_images(image_nos):
    if isinstance(image_nos, int):
        image_nos = range(image_nos)
    images = []
    for image_no in image_nos:
        image = plt.imread(DATA_DIR / f"BioID_{image_no:04}.pgm")
        images.append(image)
    return np.stack(images)


def _read_single_image_eye_annotations(img_no):
    filename = DATA_DIR / f"BioID_{img_no:04}.eye"
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
