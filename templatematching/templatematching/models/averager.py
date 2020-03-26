import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

from .utils import make_template_mass


class Averager(object):
    def __init__(self):
        self._template = None
        self._template_full = None
        self._mask = None

    def fit(self, X):
        """
        Inputs:
        -------
        train_patches (array):
            Array of shape (num_patch, patch_size) with the training semples
            (i.e. positive patches)
        n_order (int):
            The order to perform smoothing (c.f. preprocessing.m_function)
        """

        m = np.mean(X, axis=0)

        self._template_full = (m - np.mean(m)) / np.std(m)
        self._mask = make_template_mass(int(X.shape[1]/2))
        self._template = self._mask * self._template_full

    def predict(self, X, ax=None):
        """
        Inputs:
        -------
        X (array):
            Array in gray tone (2D)
        """

        conv = correlate2d(X, self._template, mode='same')
        (y, x) = np.where(conv==np.amax(conv))

        return conv, (y, x)
