import re
import numpy as np

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "images"


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
