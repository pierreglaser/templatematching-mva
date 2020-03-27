import numpy as np
from scipy.signal import fftconvolve
import functools
from scipy.integrate import dblquad


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


class ImageTransformer:
    """
    Normalize images based on Foracchia's Luminosity-Contrast
    normalization scheme
    """

    def __init__(self, wind_order=3, wind_radius=10):
        self.wind_order = wind_order
        self.wind_radius = wind_radius

    def fit(self, X, y=None):

        X = np.linspace(
            -self.wind_radius, self.wind_radius, 2 * self.wind_radius + 1
        )
        Y = np.linspace(
            -self.wind_radius, self.wind_radius, 2 * self.wind_radius + 1
        )
        x, y = np.meshgrid(X, Y)

        m_function_part = functools.partial(
            m_function, r=self.wind_radius, n_order=self.wind_order
        )

        # Compute normalizing constante
        eta = dblquad(m_function_part, -np.inf, np.inf, -np.inf, np.inf)[0]

        self.window = (
            m_function(y, x, r=self.wind_radius, n_order=self.wind_order) / eta
        )

    def transform(self, X):
        X = X.astype(np.uint8)
        transformed_X = self._normalize_img_batched(X, self.window)
        return transformed_X

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def _normalize_img_batched(self, image, window, mask=None, eps=1e-7):
        if mask is None:
            mask = np.ones((image.shape[-2], image.shape[-1]))

        window = window.reshape(1, *window.shape)
        mask = mask.reshape(1, *mask.shape)

        mask = mask.astype(int)
        mask_c_window = fftconvolve(mask, window, mode="same")
        im_squared = image ** 2

        im_mean = fftconvolve(image * mask, window, mode="same") / (
            mask_c_window + eps
        )
        im_mean_sq = fftconvolve(im_squared * mask, window, mode="same") / (
            mask_c_window + eps
        )

        std = np.sqrt(np.abs(im_mean_sq - im_mean ** 2))

        background = (1 - np.abs(image - im_mean) / (std + eps)) >= 0
        background = background.astype(int)
        mask_c_background = fftconvolve(mask * background, window, mode="same")

        im_mean = fftconvolve(
            image * mask * background, window, mode="same"
        ) / (mask_c_background + eps)
        im_mean_sq = fftconvolve(
            im_squared * mask * background, window, mode="same"
        ) / (mask_c_background + eps)

        std = np.sqrt(np.abs(im_mean_sq - im_mean ** 2))
        return np.tanh(8 * (image - im_mean) / (std + eps))
