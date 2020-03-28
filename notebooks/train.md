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
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.signal import convolve2d

from templatematching.utils import read_images, read_eye_annotations
from templatematching.models.utils import make_template_mass
from templatematching.models import Averager, R2Ridge
from templatematching.image_transformer import Normalizer
from sklearn.pipeline import make_pipeline
```

```python
def show_template_and_prediction(clf, test_images, image_no):
    convs, positions = clf.predict(images)
    conv, (x, y) = convs[image_no], positions[image_no]

    estimator = clf.steps[-1][1]
    template, mask = estimator.template, estimator._mask
    masked_template = mask * template

    f, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
    ax1.matshow(test_images[image_no], cmap="gray")
    ax1.scatter(x, y, c="r")
    ax2.matshow(conv, cmap="gray")

    ax3.matshow(template, cmap="gray")
    ax4.matshow(masked_template, cmap="gray")
```

# Average model (A) TEST

```python
num_images = 200
images, eye_annotations = read_images(num_images), read_eye_annotations(num_images)
averager_pipeline = make_pipeline(Normalizer(), Averager(patch_shape=(101, 101)))
averager_pipeline.fit(images, eye_annotations)
```

```python
show_template_and_prediction(averager_pipeline, images[:5], 0)
```

# Ridge Model (B, C)

```python
clf = R2Ridge(
    template_shape=(101, 101), splines_per_axis=(51, 51), mu=1e7,
    spline_order=3, solver="dual"
)
ridge_pipeline = make_pipeline(Normalizer(), clf)
ridge_pipeline.fit(X=images, y=eye_annotations)
```

```python
show_template_and_prediction(ridge_pipeline, images, 0)
```

# Logistic model (B, C)

```python
from templatematching.models.logistic import R2LogReg

logistic_regressor = R2LogReg(
    template_shape=(101, 101),
    splines_per_axis=(51, 51),
    mu=1e-4,
    spline_order=3,
    max_iter=50,
    random_state=10,
    verbose=1
)
logistic_pipeline = make_pipeline(Normalizer(), logistic_regressor)
logistic_pipeline.fit(X=images[:10], y=eye_annotations)
```

```python
show_template_and_prediction(logistic_pipeline, images, 0)
```

```python

```
