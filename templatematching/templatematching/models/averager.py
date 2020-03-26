import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

from .utils import make_template_mass


class Averager(object):

    def __init__(self, spline_order=2, verbose=0):
        self.spline_order = spline_order
        self.verbose = verbose
        self._template = None
        self._template_full = None
        self._mask = None

    def fit(self, X):
        """
        Inputs:
        -------
        X (array):
            Array of shape (num_patch, patch_size) with the training semples (i.e. positive patches)
        """

        m = np.mean(X, axis=0)

        self._template_full = (m - np.mean(m)) / np.std(m)
        self._mask = make_template_mass(int(X.shape[1]/2))
        self._template = self._mask * self._template_full

    def predict(self, X):
        """
        Inputs:
        -------
        X (array):
            Array in gray tone (2D)
        """

        conv = correlate2d(X, self._template, mode='same')
        (y, x) = np.where(conv==np.amax(conv))

        return conv, (y, x)
