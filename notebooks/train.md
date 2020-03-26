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

from templatematching.utils import load_patches, read_norm_img, read_pgm
from templatematching.models.utils import make_template_mass
```

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
patches, labels = load_patches(1000)
```

```python
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

ax1.matshow(patches[0], cmap='gray')
ax1.set_title('positive patch')

ax2.matshow(patches[1000], cmap='gray')
ax2.set_title('negative patch')

```

# Average model (A) TEST

```python
from templatematching.models.averager import Averager
clf = Averager(spline_order=3)
clf.fit(patches)
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

conv, (x, y) = clf.predict(image)

ax1.matshow(image, cmap='gray')
ax2.matshow(conv, cmap='gray')
ax3.matshow(img, cmap='gray')
ax3.scatter(y, x, c='r')
```

# Ridge Model (B, C)

```python
from templatematching.models import R2Ridge
clf = R2Ridge(template_shape=(51, 51), mu=0, spline_order=3)
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
image = read_norm_img(17)
img = read_pgm(17)
conv, (x, y) = clf.predict(image)


f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))
ax1.matshow(image, cmap='gray')
ax2.matshow(conv, cmap='gray');
ax3.matshow(img, cmap='gray');
ax3.scatter(y, x, c='r')
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
image = read_norm_img(17)
img = read_pgm(17)
conv, (x, y) = clf.predict(image)


f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))
ax1.matshow(image, cmap='gray')
ax2.matshow(conv, cmap='gray');
ax3.matshow(img, cmap='gray');
ax3.scatter(y, x, c='r')
```

```python

```
