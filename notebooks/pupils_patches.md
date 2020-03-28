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
from templatematching.preprocessing import PatchCreator, Normalizer
from templatematching.datasets import read_images, read_eye_annotations
```
# Test PatchCreator

```python
num_images = 100
images, eye_annotations = read_images(100), read_eye_annotations(100)
```

```python
trans = PatchCreator(patch_shape=(101, 101), neg_pos_proportion=2, random_state=1)
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
