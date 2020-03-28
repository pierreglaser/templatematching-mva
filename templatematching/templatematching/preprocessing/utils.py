import numpy as np


def mask_img(image, c=(0, 0), r=10):
    """
    Create disk-like mask for selecytong region of interest.
    The disk is centered on the image's center

    Inputs:
    -------
    image (array):
        The input image
    c (tuple):
        The center
    r (int):
        the radius of the disk

    """
    (dim_y, dim_x) = image.shape
    X = np.linspace(-dim_x / 2 + 1, dim_x / 2, dim_x)
    Y = np.linspace(-dim_y / 2 + 1, dim_y / 2, dim_y)
    x, y = np.meshgrid(X, Y)

    mask = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2) <= r

    return mask
