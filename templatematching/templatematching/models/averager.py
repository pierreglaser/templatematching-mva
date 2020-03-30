import numpy as np

from scipy.signal import correlate

from .base import PatchRegressorBase, TemplateCrossCorellatorBase
from ..preprocessing import OrientationScoreTransformer


class Averager(TemplateCrossCorellatorBase, PatchRegressorBase):
    def __init__(self, template_shape, eye="left"):
        PatchRegressorBase.__init__(self, template_shape, eye=eye)
        TemplateCrossCorellatorBase.__init__(self, template_shape=template_shape)
        self.model_name = "Averager"

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
        self._template = (m - np.mean(m)) / np.std(m)

    @TemplateCrossCorellatorBase.template.getter
    def template(self):
        if self._is_fitted:
            return self._template
        else:
            raise AttributeError("No template yet: Classifier not fitted")


class SE2Averager(TemplateCrossCorellatorBase, PatchRegressorBase):
    def __init__(
        self,
        template_shape,
        wavelet_dim,
        num_orientation_slices=12,
        batch_size=10,
        eye="left",
    ):
        PatchRegressorBase.__init__(self, template_shape, eye=eye)
        TemplateCrossCorellatorBase.__init__(self, template_shape=template_shape)
        self.model_name = "SE2Averager"
        self.batch_size = batch_size
        self._ost = OrientationScoreTransformer(
            wavelet_dim=wavelet_dim,
            num_slices=num_orientation_slices,
            batch_size=batch_size,
        )

    def predict(self, X):
        # TODO: put this method in a Mixin Class.
        X = self._ost.transform(X).imag
        template = self.template.reshape(1, *self.template.shape)
        print(X.shape)
        batch_size = min(self.batch_size, X.shape[0])

        convs = np.zeros(X.shape)

        for i in range(int(X.shape[0] / batch_size)):
            X_batch = X[i * batch_size : (i + 1) * batch_size, :, :]
            convs[i * batch_size : (i + 1) * batch_size, :, :] = correlate(
                X_batch, template, mode="same", method="fft"
            )
            positions = []

        for i in range(len(X)):
            (y, x, _) = np.unravel_index(np.argmax(convs[i]), convs[i].shape)
            positions.append([x, y])
        return convs, np.array(positions)

    def _fit_patches(self, X, y):
        """
        Inputs:
        -------
        X:
            Array of shape (num_patches, patch_shape[0], patch_shape[1])
        y:
            List: 1 if positive patch, 0 if negative
        """
        X = self._ost.fit_transform(X).imag  # can also take imag
        X = X[y == 1]  # select only positive patches
        m = np.mean(X, axis=0)
        self._template = (m - np.mean(m)) / np.std(m)

    @TemplateCrossCorellatorBase.template.getter
    def template(self):
        if self._is_fitted:
            return self._template
        else:
            raise AttributeError("No template yet: Classifier not fitted")
