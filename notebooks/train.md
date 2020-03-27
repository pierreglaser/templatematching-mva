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
%load_ext autoreload
%autoreload 2
```

```python
import os

import numpy as np
from scipy.signal import convolve2d

from templatematching.utils import read_images, read_eye_annotations, load_patches
from templatematching.models.utils import make_template_mass
```

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
num_images = 10
images, eye_annotations = read_images(num_images), read_eye_annotations(num_images)
```

# Average model (A) TEST

```python
images.shape
```

```python
from templatematching.models.averager import Averager
clf = Averager(patch_size=(51, 51))
a = clf.fit(images, eye_annotations)
```

```python
full_template, final_template = clf._template_full, clf._template

f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
ax1.matshow(full_template, cmap='gray');
ax2.matshow(final_template, cmap='gray');
```

```python
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
image = read_norm_img(17)
img = read_pgm(17)

conv, (y, x) = clf.predict(image)

ax1.matshow(image, cmap='gray')
ax2.matshow(conv, cmap='gray')
ax3.matshow(img, cmap='gray')
ax3.scatter(x, y, c='r')
```

# Ridge Model (B, C)

```python
from templatematching.models import R2Ridge
clf = R2Ridge(splines_per_axis=(51, 51), mu=0, spline_order=3, solver='dual')
clf.fit(X=patches, y=labels)
clf2 = R2Ridge(splines_per_axis=(51, 51), mu=0, spline_order=3, solver='primal')
clf2.fit(X=patches, y=labels)
```

```python
final_template = clf.reconstruct_template()
full_template = clf._template_full

f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))
ax1.matshow(clf.spline_coef.reshape(51, 51), cmap='gray')
ax2.matshow(full_template, cmap='gray');
ax3.matshow(final_template, cmap='gray');
```

```python
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
image = read_norm_img(17)
img = read_pgm(17)

conv, (y, x) = clf.predict(image)

ax1.matshow(image, cmap='gray')
ax2.matshow(conv, cmap='gray')
ax3.matshow(img, cmap='gray')
ax3.scatter(x, y, c='r')
```

# Logistic model (B, C)

```python
from templatematching.models.logistic import R2LogReg
clf = R2LogReg(template_shape=(51, 51), mu=1e-4, spline_order=3, optimizer_steps=10, random_state=10)
clf.fit(X=patches, y=labels)
```

```python
final_template = clf.reconstruct_template()
full_template = clf._template_full

f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))
ax1.matshow(clf.spline_coef.reshape(51, 51), cmap='gray')
ax2.matshow(full_template, cmap='gray');
ax3.matshow(final_template, cmap='gray');
```

```python
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
image = read_norm_img(17)
img = read_pgm(17)

conv, (y, x) = clf.predict(image)

ax1.matshow(image, cmap='gray')
ax2.matshow(conv, cmap='gray')
ax3.matshow(img, cmap='gray')
ax3.scatter(x, y, c='r')
```
