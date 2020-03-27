---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from templatematching.utils import read_pgm, read_eye_annotations, read_patch

# path to patches
path_to_positive_patches = '../patches/positive/'
path_to_negative_patches = '../patches/negative/'

```

## Create positive patches for each eye and store them 


## Create random negative patches

```python
def pos_patch(image_no, patch_size):

    # get image
    image = plt.imread('../img_normalized/img_norm_{0:04}.jpg'.format(image_no))

    # Convert to 2D array
    image = np.mean(image, axis=2)

    # Get eyes location
    left_eye, right_eye = read_eye_annotations(img_no=image_no)

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


def neg_patch(image_no, mask, patch_size, rs):

    # get image
    image = plt.imread('../img_normalized/img_norm_{0:04}.jpg'.format(image_no))

    # Convert to 2D array
    image = np.mean(image, axis=2)

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

```

```python
image_no =1
im = plt.imread('../img_normalized/img_norm_{0:04}.jpg'.format(image_no))
print(im.shape)
plt.imshow(np.mean(im, axis=2), cmap='gray')
```

```python
import os

if not os.path.exists(path_to_negative_patches):
    os.makedirs(path_to_negative_patches)
    
if not os.path.exists(path_to_positive_patches):
    os.makedirs(path_to_positive_patches)
    
if not os.path.exists(path_to_positive_patches + 'left'):
    os.makedirs(path_to_positive_patches + 'left')
    
if not os.path.exists(path_to_positive_patches + 'right'):
    os.makedirs(path_to_positive_patches + 'right')
    
    
num_images = 1521 #1521

patch_size = (101, 101)

rs = np.random.RandomState(seed=1)

# Extract positive patches
for i in range(num_images):


    left, right, mask = pos_patch(i, patch_size)

    n_patch = neg_patch(i, mask, patch_size, rs)

    # Store left patch 
    plt.imsave(path_to_positive_patches + 'left/left_patch_{0:04}.jpg'.format(i), left, cmap='gray')

    # Store right patch
    plt.imsave(path_to_positive_patches + 'right/right_patch_{0:04}.jpg'.format(i), right, cmap='gray')

    # Store negative patch
    plt.imsave(path_to_negative_patches + 'neg_patch_{0:04}.jpg'.format(i), n_patch, cmap='gray')
```

## Negative patch sample

```python
im = read_patch(0, 'negative')
plt.imshow(im, cmap='gray')
```

## Positive patches samples

```python
fig, ax = plt.subplots(1, 2)
left = read_patch(0, loc='left')
ax[0].imshow(left, cmap='gray')
ax[0].set_title('Left eye')

right = read_patch(0, loc='right')
ax[1].imshow(right, cmap='gray')
ax[1].set_title('Right eye')
```

# Test Patch_transformer 

```python
from templatematching.patch_transformer import PatchCreator
from templatematching.image_transformer import Normalizer
from templatematching.utils import read_images, read_eye_annotations
```

```python
num_images = 100
images, eye_annotations = read_images(100), read_eye_annotations(100)
```

```python
trans = PatchCreator(patch_size = (101, 101), neg_pos_proportion=2, random_state=1)
image_transformer = Normalizer()
left_eye_patches, right_eye_patches, negative_patches = trans.fit_transform(
    images, eye_annotations
)
```

```python
f, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
ax1.imshow(left_eye_patches[0])
ax2.imshow(right_eye_patches[0])
ax3.imshow(negative_patches[0])
ax4.imshow(negative_patches[1])
```
