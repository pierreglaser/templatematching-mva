import numpy as np

from abc import ABC, abstractmethod

from ..patch_transformer import PatchCreator


class PatchRegressorBase(ABC):
    def __init__(self, patch_size, **patch_creator_kwargs):
        self._patch_creator = PatchCreator(patch_size, **patch_creator_kwargs)

    def _create_patches(self, images, eye_annotations):
        patches = self._patch_creator.fit_transform(images, y=eye_annotations)
        left_patches, right_patches, negative_patches = patches
        labels = [1] * (len(left_patches) + len(right_patches))
        labels += [0] * len(negative_patches)
        all_patches = np.concatenate(
            [left_patches, right_patches, negative_patches]
        )
        return all_patches, np.array(labels)

    @abstractmethod
    def _fit_patches(X, y):
        pass

    def fit(self, X, y):
        patches, labels = self._create_patches(X, y)
        return self._fit_patches(patches, labels)
