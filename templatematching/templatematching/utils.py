import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "images"
PROCESS_DATA_DIR = Path(__file__).parent.parent.parent / "img_normalized"
PATCH_DIR = Path(__file__).parent.parent.parent / "patches"


def read_pgm(img_no, byteorder=">"):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    filename = DATA_DIR / f"BioID_{img_no:04}.pgm"
    with open(filename, "rb") as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"  # noqa
            b"(\d+)\s(?:\s*#.*[\r\n])*"  # noqa
            b"(\d+)\s(?:\s*#.*[\r\n])*"  # noqa
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)",  # noqa
            buffer,
        ).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(
        buffer,
        dtype="u1" if int(maxval) < 256 else byteorder + "u2",
        count=int(width) * int(height),
        offset=len(header),
    ).reshape((int(height), int(width)))


def read_eye_annotations(img_no):
    filename = DATA_DIR / f"BioID_{img_no:04}.eye"
    with open(filename, 'r') as f:
        lines = list(f.readlines())
    eyes_position = lines[1].strip('\n').split('\t')
    eyes_position = list(map(int, eyes_position))

    assert len(eyes_position) == 4
    left_eye_position = tuple(eyes_position[:2])
    right_eye_position = tuple(eyes_position[2:4])
    return left_eye_position, right_eye_position


def read_patch(img_no, pos_neg='positive', loc='left'):
    """
    Inputs:
    -------
    img_no (int):
        The patch id
    pos_neg(str):
        Should be negative or postive for choice in patch
    loc (string):
        Should be 'left' or 'right' for eye location

    Returns a specific positive/negative patch
    """

    assert(pos_neg == 'positive' or pos_neg == 'negative')
    assert(loc == 'left' or loc == 'right')

    if pos_neg == 'positive':
        filename = PATCH_DIR / f'{pos_neg}' / f"{loc}" / f"{loc}_patch_{img_no:04}.jpg"
        patch = plt.imread(filename)

        # Put in gray scale
        patch = np.mean(patch, axis=2)

        return patch

    filename = PATCH_DIR / 'negative' / f"neg_patch_{img_no:04}.jpg"
    patch = plt.imread(filename)

    # Put in gray scale
    patch = np.mean(patch, axis=2)

    return patch


def load_patches(num_patches, with_labels=True):
    pos_patches = np.zeros((num_patches, 101, 101))
    pos_labels = np.ones((num_patches, 1))

    neg_patches = np.zeros((num_patches, 101, 101))
    neg_labels = np.zeros((num_patches, 1))

    for i in range(num_patches):
        pos_patch = read_patch(i, loc='left', pos_neg="positive")
        neg_patch = read_patch(i, loc='left', pos_neg="negative")

        pos_patches[i] = pos_patch
        neg_patches[i] = neg_patch

    all_patches = np.vstack((pos_patches, neg_patches))
    all_labels = np.vstack((pos_labels, neg_labels))
    return all_patches, all_labels


def read_norm_img(img_no):

    filename = PROCESS_DATA_DIR / f'img_norm_{img_no:04}.jpg'

    image = plt.imread(filename)

    image = np.mean(image, axis=2)

    return image
