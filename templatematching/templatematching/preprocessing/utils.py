import numpy as np


def m_function(x, y, r, n_order=1):
    r"""
    The windowing function a defined in the article.

    Carreful: The output is not normalized such that
    \\int_{\mathbb{R}^2} m(x, y) dx dy = 1
    ---------
    To find the constante call 'dblquad' from scipy.integrate

    Inputs:
    -------
    x (int):
        x-coordinate with (0, 0) in center of image
    y (int):
        y-coordinate with (0, 0) in center of image
    r (float):
        the disk radius
    n_order (int):
        the order of the sum (c.f. paper)
    """

    res = 0

    norm_sq = x * x + y * y
    s = 2 * r * r / (1 + 2 * n_order)

    for i in range(n_order):

        res += np.exp(-norm_sq / s) * (norm_sq / s) ** i / np.math.factorial(i)

    return res


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
