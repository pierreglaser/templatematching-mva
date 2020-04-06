import numpy as np
import functools

from math import ceil

from scipy.signal import fftconvolve
from scipy.integrate import dblquad
from joblib import Parallel, delayed

from .utils import m_function
from ..utils import tqdm


def _normalize_images_batched(
    image, window, batch_size=50, mask=None, eps=1e-7, n_jobs=1
):
    if mask is None:
        mask = np.ones((image.shape[-2], image.shape[-1]))

    window = window.reshape(1, *window.shape)
    mask = mask.reshape(1, *mask.shape)

    mask = mask.astype(int)
    mask_c_window = fftconvolve(mask, window, mode="same")

    images_normalized = Parallel(prefer="threads", n_jobs=n_jobs)(
        delayed(_normalize_batch)(
            image[i * batch_size: (i + 1) * batch_size, :, :],  # noqa
            window,
            mask,
            eps,
            mask_c_window,
        )
        for i in tqdm(range(ceil(image.shape[0] / batch_size)),
                      desc="normalization")
    )
    return np.concatenate(images_normalized)


def _normalize_batch(images, window, mask, eps, mask_c_window=None):
    im_squared = images ** 2

    im_mean = fftconvolve(images * mask, window, mode="same",) / (
        mask_c_window + eps
    )
    im_mean_sq = fftconvolve(im_squared * mask, window, mode="same") / (
        mask_c_window + eps
    )

    std = np.sqrt(np.abs(im_mean_sq - im_mean ** 2))

    background = (1 - np.abs(images - im_mean) / (std + eps)) >= 0
    background = background.astype(int)
    mask_c_background = fftconvolve(mask * background, window, mode="same")

    im_mean = fftconvolve(images * mask * background, window, mode="same",) / (
        mask_c_background + eps
    )
    im_mean_sq = fftconvolve(
        im_squared * mask * background, window, mode="same"
    ) / (mask_c_background + eps)

    std = np.sqrt(np.abs(im_mean_sq - im_mean ** 2))
    return np.tanh(8 * (images - im_mean) / (std + eps))


class Normalizer:
    """
    Normalize images based on Foracchia's Luminosity-Contrast
    normalization scheme
    """

    def __init__(self, wind_order=3, wind_radius=10, batch_size=50, n_jobs=1):
        self.wind_order = wind_order
        self.wind_radius = wind_radius
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def fit(self, X, y=None):

        X = np.linspace(-self.wind_radius, self.wind_radius, 2 * self.wind_radius + 1)
        Y = np.linspace(-self.wind_radius, self.wind_radius, 2 * self.wind_radius + 1)
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
        # XXX: we should not cast as int8 as _normalize_images_batched
        # cause int8 overflow. But somehow the preprocessing looks
        # better in this case...
        X = X.astype(np.uint8)
        transformed_X = _normalize_images_batched(
            X,
            self.window,
            min(X.shape[0], self.batch_size),
            n_jobs=self.n_jobs,
        )
        return transformed_X

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)
