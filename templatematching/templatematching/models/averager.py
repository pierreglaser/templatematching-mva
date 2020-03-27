import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

from .utils import make_template_mass
from .base import PatchRegressorBase


class Averager(PatchRegressorBase):
    def __init__(self, patch_size):
        super().__init__(patch_size)
        self.model_name = "Averager"
        self._template = None
        self._template_full = None
        self._mask = None

    def _fit_patches(self, X, y):
        """
        Inputs:
        -------
        X:
            Array of shape (num_patches, patch_shape[0], patch_shape[1])
        y:
            List: 1 if positive patch, 0 if negative
        n_order (int):
            The order to perform smoothing (c.f. preprocessing.m_function)
        """
        X = X[y == 1]  # select only positive patches

        m = np.mean(X, axis=0)

        self._template_full = (m - np.mean(m)) / np.std(m)
        self._mask = make_template_mass(int(X.shape[1] / 2))
        self._template = self._mask * self._template_full

    def predict(self, X, ax=None):
        """
        Inputs:
        -------
        X (array):
            Array in gray tone (2D)
        """

        conv = correlate2d(X, self._template, mode="same")
        (y, x) = np.where(conv == np.amax(conv))

        return conv, (y, x)

    def score(self, X, y, radius_criteria=50):

        num_sample = X.shape[0]

        score = 0

        for i in range(num_sample):
            image = X[i]
            _, (pred_y, pred_x) = self.predict(image)

            true_x, true_y = y[i][0], y[i][1]

            if np.sqrt(
                (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2
            ) < np.sqrt(radius_criteria):

                score += 1

        total_score = np.round(score / num_sample * 100, 2)

        print(f"Score was computed on {num_sample} samples: \n")
        print(f"Model {self.model_name} accuracy: {total_score} %")
