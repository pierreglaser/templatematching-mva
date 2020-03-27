import numpy as np
from templatematching.utils import read_eye_annotations


def _pos_patch(image, image_no, patch_size):

    left_eye, right_eye = read_eye_annotations(img_no=int(image_no))

    # Define offset 
    Nx, Ny = int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2)

    # Define mask
    mask = np.ones(image.shape)


    # Perform padding
    pad_1, pad_2 = - min(left_eye[1] - Ny, 0), - image.shape[0] + max(left_eye[1] + Ny + 1, image.shape[0])
    pad_3, pad_4 = - min(left_eye[0] - Nx, 0), - image.shape[1] + max(left_eye[0] + Ny + 1, image.shape[1])

    # Create left patch
    patch_left =  image[max(left_eye[1] - Ny, 0):min(left_eye[1] + Ny + 1, image.shape[0]), max(left_eye[0] - Nx, 0): min(left_eye[0] + Nx + 1, image.shape[1])]
    
    # Remove location of left patch
    mask[max(left_eye[1] - Ny, 0):min(left_eye[1] + Ny + 1, image.shape[0]), max(left_eye[0] - Nx, 0): min(left_eye[0] + Nx + 1, image.shape[1])] = 0

    # Pad the patch
    patch_left = np.pad(patch_left, ((pad_1, pad_2), (pad_3, pad_4)))

    assert(patch_left.shape == patch_size)    


    # Perform padding
    pad_1, pad_2 = - min(right_eye[1] - Ny, 0), - image.shape[0] + max(right_eye[1] + Ny + 1, image.shape[0])
    pad_3, pad_4 = - min(right_eye[0] - Nx, 0), - image.shape[1] + max(right_eye[0] + Ny + 1, image.shape[1])


    # Create right patch
    patch_right =  image[max(right_eye[1] - Ny, 0):min(right_eye[1] + Ny + 1, image.shape[0]), max(right_eye[0] - Nx, 0): min(right_eye[0] + Nx + 1, image.shape[1])]

    # Remove location of right patch
    mask[max(right_eye[1] - Ny, 0):min(right_eye[1] + Ny + 1, image.shape[0]), max(right_eye[0] - Nx, 0): min(right_eye[0] + Nx + 1, image.shape[1])] = 0


    # Pad the patch
    patch_right = np.pad(patch_right, ((pad_1, pad_2), (pad_3, pad_4)))

    assert(patch_right.shape == patch_size)

    return patch_left, patch_right, mask


def _neg_patch(image, image_no, mask, patch_size, rs):


    assert(mask.shape == image.shape)

    Nx, Ny = int((patch_size[0] - 1) / 2), int((patch_size[1] - 1) / 2)

    v = np.argwhere(mask==1)

    center_id = rs.randint(0, len(v))

    center = v[center_id]

    # Perform padding
    pad_1, pad_2 = - min(center[0] - Ny, 0), - image.shape[0] + max(center[0] + Ny + 1, image.shape[0])
    pad_3, pad_4 = - min(center[1] - Nx, 0), - image.shape[1] + max(center[1] + Ny + 1, image.shape[1])

    patch =  image[max(center[0] - Ny, 0):min(center[0] + Ny + 1, image.shape[0]), max(center[1] - Nx, 0): min(center[1] + Nx + 1, image.shape[1])]
    

    # Pad the patch
    patch = np.pad(patch, ((pad_1, pad_2), (pad_3, pad_4)))

    assert(patch.shape == patch_size)

    return patch



class PatchTranformer:
    """
    Creation of positive and negative patch
    from images
    """

    def __init__(
        self,
        patch_size,
        neg_pos_proportion=1,
        random_state=None
    ):
        """
        Inputs:
        -------

        patch_size (tuple(int)): 
            Patches size
        neg_pos_proportion (float):
            The ratio negative / positive patches
        """
        self.rs = np.random.RandomState(random_state)
        self.patch_size = patch_size
        self.neg_pos_prop = neg_pos_proportion

    def fit(self, X, y=None):
        """
        Inputs:
        -------

        X (array):
            The images previously normalized
        y (array):
            The corresponding images number
        """
        self.img_no = y

    def transform(self, X):

        pos_patches_left = []
        pos_patches_right = []
        neg_patches = [] 

        for i in range(X.shape[0]):

            left, right, mask = _pos_patch(X[i], self.img_no[i], self.patch_size)

            pos_patches_left.append(left)
            pos_patches_right.append(right)
            
            for _ in range(self.neg_pos_prop):
                neg_patches.append(_neg_patch(X[i], self.img_no[i], mask, self.patch_size, self.rs))
        
        return np.stack(pos_patches_left, axis=0), np.stack(pos_patches_right, axis=0), np.stack(neg_patches, axis=0)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)