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
import time

from templatematching.datasets import load_facesdb
from templatematching.preprocessing import (
    Normalizer,
    PatchCreator,
    OrientationScoreTransformer,
)
```

```python
images, eye_annotations = load_facesdb()
```

```python
t0 = time.time()
normalizer = Normalizer(n_jobs=-1)
normalized_images = normalizer.fit_transform(images[:100])
print(f'normalization total time {time.time() - t0}')
```

```python
t0 = time.time()
trans = PatchCreator(patch_shape=(101, 101), neg_pos_proportion=2, random_state=1, n_jobs=-1)
left_eye_patches, right_eye_patches, negative_patches = trans.fit_transform(
    normalized_images, eye_annotations
)
print(f'patch creation total time {time.time() - t0}')
```

```python
t0 = time.time()
transformer = OrientationScoreTransformer(wavelet_dim=501, num_slices=12, n_jobs=-1)
cx = transformer.fit_transform(images[:10])
print(f'orientation score creation total time {time.time() - t0}')
```
