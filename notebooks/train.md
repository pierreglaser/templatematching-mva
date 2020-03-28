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

from sklearn.pipeline import make_pipeline

from templatematching.datasets import read_images, read_eye_annotations
from templatematching.models.utils import make_template_mass
from templatematching.models import Averager, R2Ridge, R2LogReg
from templatematching.preprocessing import Normalizer
```

```python
def show_template_and_prediction(clf, test_images, image_no):
    convs, positions = clf.predict(images)
    conv, (x, y) = convs[image_no], positions[image_no]

    est_name, estimator = clf.steps[-1]
    template, mask = estimator.template, estimator._mask
    if est_name == 'se2ridge':
        template = template[:, :, 0]
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

```python
num_test_samples = 5
score = averager_pipeline.score(
    images[:num_test_samples], eye_annotations[:num_test_samples]
)
print(f'score (from {num_test_samples} samples): {score:.3f}')
```

# Ridge Model (B, C)

```python
clf = R2Ridge(
    template_shape=(101, 101), splines_per_axis=(51, 51),
    mu=1e7, spline_order=3, solver="dual"
)
ridge_pipeline = make_pipeline(Normalizer(), clf)
ridge_pipeline.fit(X=images, y=eye_annotations)
```

```python
show_template_and_prediction(ridge_pipeline, images, 0)
```

```python
num_test_samples = 5
score = ridge_pipeline.score(
    images[:num_test_samples], eye_annotations[:num_test_samples]
)
print(f'score (from {num_test_samples} samples): {score:.3f}')
```

# Logistic model (B, C)

```python
logistic_regressor = R2LogReg(
    template_shape=(101, 101),
    splines_per_axis=(51, 51),
    mu=1e-4,
    spline_order=3,
    max_iter=50,
    random_state=10,
    tol=1e-6,
    verbose=1
)
logistic_pipeline = make_pipeline(Normalizer(), logistic_regressor)
logistic_pipeline.fit(X=images[:10], y=eye_annotations)
```

```python
show_template_and_prediction(logistic_pipeline, images, 0)
```

```python
num_test_samples = 5
score = logistic_pipeline.score(
    images[:num_test_samples], eye_annotations[:num_test_samples]
)
print(f'score (from {num_test_samples} samples): {score:.3f}')
```

```python
from templatematching.models.linear import SE2Ridge
```

```python
%pdb
```

```python
num_images = 10
images, eye_annotations = read_images(num_images), read_eye_annotations(num_images)
se2_pipeline = make_pipeline(
    Normalizer(),
    SE2Ridge(template_shape=(101, 101), splines_per_axis=(51, 51, 4),
             wavelet_dim=21, num_orientation_slices=4,
             mu=1e7, spline_order=3, solver="dual")
)
se2_pipeline.fit(images, eye_annotations)
```

```python
show_template_and_prediction(se2_pipeline, images, 0)
```
