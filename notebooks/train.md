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
from sklearn.model_selection import train_test_split

from templatematching.datasets import read_images, read_eye_annotations
from templatematching.models.utils import make_template_mass
from templatematching.models import (
    Averager,
    SE2Averager,
    R2Ridge,
    R2LogReg,
    SE2Ridge,
    SE2LogReg
)

from templatematching.preprocessing import Normalizer
```

```python
def show_template_and_prediction(clf, test_images, image_no):
    convs, positions = clf.predict(test_images)
    conv, (x, y) = convs[image_no], positions[image_no]

    est_name, estimator = clf.steps[-1]
    template, mask = estimator.template, estimator._mask
    if est_name == 'se2ridge' or est_name == 'se2logreg' or est_name == 'se2averager':
        template = template[:, :, 0]
        conv = conv[:, :, 0]
    masked_template = mask * template

    f, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
    ax1.matshow(test_images[image_no], cmap="gray")
    ax1.scatter(x, y, c="r")
    ax2.matshow(conv, cmap="gray")

    ax3.matshow(template, cmap="gray")
    ax4.matshow(masked_template, cmap="gray")
```

```python
num_images = 500
random_state = 10
images, eye_annotations = read_images(num_images), read_eye_annotations(num_images)

X_train, X_test, y_train, y_test = train_test_split(images, eye_annotations, train_size=0.8, shuffle=True, random_state=random_state)
```

```python
print(f'Number of training samples: {np.round(X_train.shape[0], 2)}')
print(f'Number of test samples: {np.round(X_test.shape[0], 2)}')
```

# Average model (A) TEST

```python
averager_pipeline = make_pipeline(Normalizer(), Averager(template_shape=(101, 101), eye='left'))
averager_pipeline.fit(X_train, y_train)
```

```python
show_template_and_prediction(averager_pipeline, X_test, 0)
```

```python
score = averager_pipeline.score(
    X_test, y_test
)
print(f'score (from {X_test.shape[0]} samples): {score:.3f}')
```

# Ridge Model (B, C, D, E)

```python
clf = R2Ridge(
    template_shape=(101, 101),
    splines_per_axis=(51, 51),
    mu=0,
    lbd=1e-0,
    spline_order=3,
    solver="dual",
    random_state=random_state
)
ridge_pipeline = make_pipeline(Normalizer(), clf)
ridge_pipeline.fit(X=X_train, y=y_train)
```

```python
show_template_and_prediction(ridge_pipeline, X_test, 0)
```

```python
score = ridge_pipeline.score(
    X_test, y_test
)
print(f'score (from {X_test.shape[0]} samples): {score:.3f}')
```

# Logistic model (B, C, D, E)

```python
logistic_regressor = R2LogReg(
    template_shape=(101, 101),
    splines_per_axis=(51, 51),
    mu=1e-1,
    lbd=1e-0,
    spline_order=3,
    max_iter=50,
    random_state=random_state,
    tol=1e-6,
    early_stopping=10,
    verbose=1
)
logistic_pipeline = make_pipeline(Normalizer(), logistic_regressor)
logistic_pipeline.fit(X=X_train, y=y_train)
```

```python
show_template_and_prediction(logistic_pipeline, X_test, 0)
```

```python
score = logistic_pipeline.score(
    X_test, y_test
)
print(f'score (from {X_test.shape[0]} samples): {score:.3f}')
```

# SE2 Averager 

```python
from templatematching.models.averager import SE2Averager
```

```python
se2_avg_pipeline = make_pipeline(
    Normalizer(),
    SE2Averager(template_shape=(101, 101),
             wavelet_dim=21,
             num_orientation_slices=12,
             batch_size=10,
             eye='left'
            )
)
se2_avg_pipeline.fit(X_train, y_train)
```

```python
show_template_and_prediction(se2_avg_pipeline, X_test, 0)
```

```python
score = se2_avg_pipeline.score(
    X_test, y_test
)
print(f'score (from {X_test.shape[0]} samples): {score:.3f}')
```

# SE2 Ridge (B, C, D, E)

```python
se2_pipeline = make_pipeline(
    Normalizer(),
    SE2Ridge(template_shape=(101, 101),
             splines_per_axis=(51, 51, 4),
             wavelet_dim=21,
             num_orientation_slices=4,
             mu=0,
             lbd=0,
             Dxi=1,
             Deta=0,
             Dtheta=1e-2,
             spline_order=3,
             solver="dual",
             random_state=random_state)
)
se2_pipeline.fit(X_train, y_train)
```

```python
show_template_and_prediction(se2_pipeline, X_test, 0)
```

```python
score = se2_pipeline.score(
    X_test, y_test
)
print(f'score (from {X_test.shape[0]} samples): {score:.3f}')
```

```python
from templatematching.models.logistic import SE2LogReg
```

```python
se2_pipeline = make_pipeline(
    Normalizer(),
    SE2LogReg(template_shape=(101, 101),
              splines_per_axis=(51, 51, 4),
              wavelet_dim=21,
              num_orientation_slices=4,
              batch_size=10,
              max_iter=50,
              tol=1e-6,
              early_stopping=10,
              mu=0,
              lbd=0,
              Dxi=1,
              Deta=0,
              Dtheta=1e-2,
              spline_order=3,
              random_state=random_state,
              verbose=1)
)
se2_pipeline.fit(X_train, y_train)
```

```python
show_template_and_prediction(se2_pipeline, X_test, 0)
```

```python
score = se2_pipeline.score(
    X_test, y_test
)
print(f'score (from {X_test.shape[0]} samples): {score:.3f}')
```

```python

```
