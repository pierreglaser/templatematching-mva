---
jupyter:
  jupytext:
    formats: ipynb,md
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
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
%load_ext autoreload
%autoreload 2
```

### Normalization 

```python
from templatematching.datasets import load_facesdb
from templatematching.preprocessing import Normalizer, PatchCreator
```

```python
# we first load the images into memory
images, eye_annotations = load_facesdb()
```

```python
normalizer = Normalizer(n_jobs=-1)
normalized_images = normalizer.fit_transform(images[:500])

f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
ax1.imshow(images[0], cmap='gray')
ax1.set_title('original image')
ax2.imshow(normalized_images[0], cmap='gray')
ax2.set_title('normalized image')
```

### Patch creation

```python
trans = PatchCreator(patch_shape=(101, 101), neg_pos_proportion=2, random_state=1, n_jobs=-1)
left_eye_patches, right_eye_patches, negative_patches = trans.fit_transform(
    normalized_images, eye_annotations
)

f, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(15, 10))
ax1.matshow(left_eye_patches[0], cmap='gray')
ax2.matshow(right_eye_patches[0], cmap='gray')
ax3.matshow(negative_patches[0], cmap='gray')
ax4.matshow(negative_patches[1], cmap='gray')
```

### Orientation scores

```python
from templatematching.datasets import make_circle, make_cross
from templatematching.preprocessing import OrientationScoreTransformer
```

```python
# these parameters take ~1 minute to fit to have a nice visualisation, 
# scale down the size of the image/the patch to increase speed.
geom_images = np.stack([make_circle(201), make_cross(201)])
transformer = OrientationScoreTransformer(wavelet_dim=501, num_slices=12)
oriented_circle, oriented_cross  = transformer.fit_transform(geom_images)

f, axs = plt.subplots(ncols=4, figsize=(20, 5))
for ax in axs:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
axs[0].matshow(transformer._cake_slices[0])

# zoom a little bit on the wavelet
# zoomed_slice = slice(int(3*transformer.wavelet_dim/8),int(5*transformer.wavelet_dim/8))
zoomed_slice = slice(int(3.75*transformer.wavelet_dim/8),int(4.25*transformer.wavelet_dim/8))

axs[1].matshow(transformer._wavelets[0].imag[zoomed_slice, zoomed_slice])
axs[2].matshow(geom_images[0])
axs[3].matshow(oriented_circle.imag[:, :, 0])
```

### Train a model

```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from templatematching.models import SE2Ridge, R2Ridge
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
random_state = 10
num_images = 500
X_train, X_test, y_train, y_test = train_test_split(
    images[:num_images],
    eye_annotations[:num_images],
    train_size=0.8,
    shuffle=True,
    random_state=random_state,
)

print(f"Number of training samples: {np.round(X_train.shape[0], 2)}")
print(f"Number of test samples: {np.round(X_test.shape[0], 2)}")
```

```python
ridge_model = R2Ridge(
    template_shape=(101, 101), splines_per_axis=(51, 51), spline_order=3,
    batch_size=1, mu=0.5 * 1e-3, lbd=0.5 * 1e-4, solver="dual",
    random_state=random_state, n_jobs=4
)
normalizer = Normalizer(n_jobs=4, batch_size=3)
```

```python
r2_ridge_pipeline = make_pipeline(normalizer, ridge_model)
r2_ridge_pipeline = r2_ridge_pipeline.fit(X_train, y_train)
```

```python
normalizer = Normalizer(batch_size=3, n_jobs=4)
se2_ridge_model = SE2Ridge(
    template_shape=(101, 101), splines_per_axis=(51, 51, 4),  # dimesions
    wavelet_dim=21, num_orientation_slices=4, # orientation scores
    mu=0.5 * 1e-3, lbd=0.5 * 1e-4, Dxi=1, Deta=0, Dtheta=1e-2,  # regularization
    solver="primal", n_jobs=4, batch_size=3, # performance
    spline_order=3, random_state=random_state # misc
)
```

```python
se2_pipeline = make_pipeline(normalizer, se2_ridge_model)
se2_pipeline = se2_pipeline.fit(X_train[:100], y_train[:100])
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

```python

```
