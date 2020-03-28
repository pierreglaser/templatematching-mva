import numpy as np

from .utils import make_template_mass
from .base import PatchRegressorBase, TemplateCrossCorellatorBase


class Averager(TemplateCrossCorellatorBase, PatchRegressorBase):
    def __init__(self, patch_shape, eye="left"):
        PatchRegressorBase.__init__(self, patch_shape, eye=eye)
        TemplateCrossCorellatorBase.__init__(self, template_shape=patch_shape)
        self.model_name = "Averager"
        self._mask = None
        self._is_fitted = None

    def _fit_patches(self, X, y):
        """
        Inputs:
        -------
        X:
            Array of shape (num_patches, patch_shape[0], patch_shape[1])
        y:
            List: 1 if positive patch, 0 if negative
        """
        X = X[y == 1]  # select only positive patches
        m = np.mean(X, axis=0)

        self._mask = make_template_mass(int(X.shape[1] / 2))
        self._template = (m - np.mean(m)) / np.std(m)
        self._masked_template = self._mask * self._template

    @TemplateCrossCorellatorBase.template.getter
    def template(self):
        if self._is_fitted:
            return self._template
        else:
            raise AttributeError("No template yet: Classifier not fitted")
