import numpy as np


class PatchCreator:
    """
    Creation of positive and negative patch
    from images
    """

    def __init__(self, patch_shape, neg_pos_proportion=1, random_state=None):
        """
        Inputs:
        -------

        patch_shape (tuple(int)):
            Patches shape
        neg_pos_proportion (float):
            The ratio negative / positive patches
        random_state (int):
            Random see, intervenes only for negative patch creation
        """
        self.rs = np.random.RandomState(random_state)
        self.patch_shape = patch_shape
        self.neg_pos_prop = neg_pos_proportion
        self._eye_locations = None

    def fit(self, X, y=None):
        """
        Inputs:
        -------

        X (array):
            The images previously normalized
        y (array):
            The eye locations
        """
        self._eye_locations = y

    def transform(self, X):

        pos_patches_left = []
        pos_patches_right = []
        neg_patches = []

        for i in range(X.shape[0]):
            left_eye_pos, right_eye_pos = self._eye_locations[i]
            # by convention, left eye is first item of the tuple
            left, right, mask = self._create_positive_patch_from_image(
                X[i], left_eye_pos, right_eye_pos
            )

            pos_patches_left.append(left)
            pos_patches_right.append(right)

            for _ in range(self.neg_pos_prop):
                patch = self._create_negative_patch_from_image(X[i], mask)
                neg_patches.append(patch)

        return (
            np.stack(pos_patches_left, axis=0),
            np.stack(pos_patches_right, axis=0),
            np.stack(neg_patches, axis=0),
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def _create_positive_patch_from_image(self, image, left_eye_pos, right_eye_pos):

        # Define offset
        Nx = int((self.patch_shape[0] - 1) / 2)
        Ny = int((self.patch_shape[1] - 1) / 2)
        # Define mask
        mask = np.ones(image.shape)

        # Perform padding
        pad_1, pad_2 = (
            -min(left_eye_pos[1] - Ny, 0),
            -image.shape[0] + max(left_eye_pos[1] + Ny + 1, image.shape[0]),
        )
        pad_3, pad_4 = (
            -min(left_eye_pos[0] - Nx, 0),
            -image.shape[1] + max(left_eye_pos[0] + Ny + 1, image.shape[1]),
        )

        # Create left patch
        patch_left = image[
            max(left_eye_pos[1] - Ny, 0) : min(
                left_eye_pos[1] + Ny + 1, image.shape[0]
            ),
            max(left_eye_pos[0] - Nx, 0) : min(
                left_eye_pos[0] + Nx + 1, image.shape[1]
            ),
        ]

        # Remove location of left patch
        mask[
            max(left_eye_pos[1] - Ny, 0) : min(
                left_eye_pos[1] + Ny + 1, image.shape[0]
            ),
            max(left_eye_pos[0] - Nx, 0) : min(
                left_eye_pos[0] + Nx + 1, image.shape[1]
            ),
        ] = 0

        # Pad the patch
        patch_left = np.pad(patch_left, ((pad_1, pad_2), (pad_3, pad_4)))

        assert patch_left.shape == self.patch_shape

        # Perform padding
        pad_1, pad_2 = (
            -min(right_eye_pos[1] - Ny, 0),
            -image.shape[0] + max(right_eye_pos[1] + Ny + 1, image.shape[0]),
        )
        pad_3, pad_4 = (
            -min(right_eye_pos[0] - Nx, 0),
            -image.shape[1] + max(right_eye_pos[0] + Ny + 1, image.shape[1]),
        )

        # Create right patch
        patch_right = image[
            max(right_eye_pos[1] - Ny, 0) : min(
                right_eye_pos[1] + Ny + 1, image.shape[0]
            ),
            max(right_eye_pos[0] - Nx, 0) : min(
                right_eye_pos[0] + Nx + 1, image.shape[1]
            ),
        ]

        # Remove location of right patch
        mask[
            max(right_eye_pos[1] - Ny, 0) : min(
                right_eye_pos[1] + Ny + 1, image.shape[0]
            ),
            max(right_eye_pos[0] - Nx, 0) : min(
                right_eye_pos[0] + Nx + 1, image.shape[1]
            ),
        ] = 0

        # Pad the patch
        patch_right = np.pad(patch_right, ((pad_1, pad_2), (pad_3, pad_4)))

        assert patch_right.shape == self.patch_shape

        return patch_left, patch_right, mask

    def _create_negative_patch_from_image(self, image, mask):
        assert mask.shape == image.shape

        Nx = int((self.patch_shape[0] - 1) / 2)
        Ny = int((self.patch_shape[1] - 1) / 2)

        v = np.argwhere(mask == 1)
        center_id = self.rs.randint(0, len(v))
        center = v[center_id]

        # Perform padding
        pad_1, pad_2 = (
            -min(center[0] - Ny, 0),
            -image.shape[0] + max(center[0] + Ny + 1, image.shape[0]),
        )
        pad_3, pad_4 = (
            -min(center[1] - Nx, 0),
            -image.shape[1] + max(center[1] + Ny + 1, image.shape[1]),
        )
        patch = image[
            max(center[0] - Ny, 0) : min(center[0] + Ny + 1, image.shape[0]),
            max(center[1] - Nx, 0) : min(center[1] + Nx + 1, image.shape[1]),
        ]

        # Pad the patch
        patch = np.pad(patch, ((pad_1, pad_2), (pad_3, pad_4)))

        assert patch.shape == self.patch_shape

        return patch
