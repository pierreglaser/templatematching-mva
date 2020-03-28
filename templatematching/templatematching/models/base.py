import numpy as np

from abc import ABC, abstractmethod

from scipy.signal import correlate

from ..patch_transformer import PatchCreator


class TemplateCrossCorellatorBase(ABC):
    def __init__(self, template_shape):
        self.template_shape = template_shape
        self._template = None
        self._masked_template = None

    @property
    @abstractmethod
    def template(self):
        pass

    def predict(self, X):
        template = self.template.reshape(1, *self.template.shape)
        convs = correlate(X, template, mode="same", method="fft")
        positions = []
        for i in range(len(X)):
            (y, x) = np.unravel_index(np.argmax(convs[0]), convs[0].shape)
            positions.append([x, y])
        return convs, np.array(positions)

    def score(self, X, y, radius_criteria=50):
        num_sample = X.shape[0]
        score = 0

        for i in range(num_sample):
            image = X[i]
            _, (pred_y, pred_x) = self.predict(image)
            true_x, true_y = y[i][0], y[i][1]

            dist_to_location = np.sqrt(
                (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2
            )
            if dist_to_location < np.sqrt(radius_criteria):
                score += 1

        total_score = np.round(score / num_sample * 100, 2)
        print(f"Score was computed on {num_sample} samples: \n")
        print(f"Model {self.model_name} accuracy: {total_score} %")
        return total_score


class SplineRegressorBase(TemplateCrossCorellatorBase):
    def __init__(self, template_shape, splines_per_axis, spline_order=2):
        TemplateCrossCorellatorBase.__init__(self, template_shape)
        self.splines_per_axis = splines_per_axis
        self.spline_order = spline_order
        self._S = None

    @abstractmethod
    def _create_s_matrix(self):
        pass

    @abstractmethod
    def _check_params(self, X):
        pass

    @abstractmethod
    def _get_dims(self, X):
        pass


class PatchRegressorBase(ABC):
    def __init__(self, patch_shape, eye="left", **patch_creator_kwargs):
        assert eye in ["left", "right"]
        self._patch_creator = PatchCreator(patch_shape, **patch_creator_kwargs)
        self.eye = eye

    def _create_patches(self, images, eye_annotations):
        patches = self._patch_creator.fit_transform(images, y=eye_annotations)
        left_patches, right_patches, negative_patches = patches
        assert len(left_patches) == len(right_patches)
        labels = [1] * len(left_patches) + [0] * len(negative_patches)

        if self.eye == "left":
            all_patches = np.concatenate([left_patches, negative_patches])
        else:
            all_patches = np.concatenate([right_patches, negative_patches])
        return all_patches, np.array(labels)

    @abstractmethod
    def _fit_patches(self, X, y):
        pass

    def fit(self, X, y):
        patches, labels = self._create_patches(X, y)
        self._fit_patches(patches, labels)
        self._is_fitted = True
        return self
