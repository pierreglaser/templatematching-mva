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

import matplotlib.pyplot as plt
%matplotlib inline

from templatematching.datasets import read_images
from templatematching.preprocessing import Normalizer
```


# Example Using Normalizer

```python
images = read_images(1)
normalizer = Normalizer()
normalized_images = normalizer.fit_transform(images)

f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax1.imshow(images[0], cmap='gray')
ax1.set_title('original image')
ax2.imshow(normalized_images[0], cmap='gray')
ax2.set_title('normalized image')
```
