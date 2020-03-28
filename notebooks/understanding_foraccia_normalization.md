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

This is a WIP notebook dedicated at better understanding the Foraccia image normalization that we use in the Image Transformer.

```python
%load_ext autoreload
%autoreload 2
```

```python
import functools

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.integrate import dblquad
from scipy.signal import fftconvolve

from templatematching.datasets import read_images
from templatematching.preprocessing.utils import m_function
from templatematching.preprocessing.image_transformer import _normalize_img_batched
```

```python
image = read_images([17])[0]
image_float = image.astype(np.float64)
```

```python
r = 10

X = np.linspace(-r, r, 2 * r + 1)
Y = np.linspace(-r, r, 2 * r + 1)
x, y = np.meshgrid(X, Y)

n_order = 1

disk_window = np.sqrt(x ** 2 + y ** 2) < r
m_function_part = functools.partial(m_function, r=r, n_order=n_order)
eta = dblquad(m_function_part, - np.inf, np.inf, -np.inf, np.inf)[0]
window = m_function(y, x, r=r, n_order=n_order) / eta
```

```python
ee = (np.round(127+ 127*_normalize_img_batched(image_float[np.newaxis, :, :], window)))[0]

f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
ax1.imshow(image, cmap='gray')
ax2.imshow(ee, cmap='gray')
```

```python
#Js = 100
window_two = np.ones(image_float.shape)
window_two /= window_two.sum()
nim = _normalize_img_batched(image_float[np.newaxis, :, :]/256, window_two)[0]
```

```python
plt.imshow(nim, cmap='gray')
```

```python
eps = 1e-7
mask = np.ones((image.shape[0], image.shape[1]))
im_mean = fftconvolve(image, window, mode="same") / (
    fftconvolve(mask, window, mode="same") + eps
)
```

```python
im_mean = fftconvolve(image_float, window_two, mode='same')
im_meansq = fftconvolve(image_float ** 2, window_two, mode='same')
std = np.sqrt(np.abs(im_meansq - im_mean**2))
```

```python
plt.imshow(1*(np.abs((image_float - im_mean))/(std + eps)>=1))
```
