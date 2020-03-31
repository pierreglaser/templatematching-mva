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

We assume the images are located in the the `images` folder, itself located at the root folder of the project.

```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

```python
from templatematching.datasets import make_circle, make_cross
from templatematching.preprocessing import OrientationScoreTransformer
```

```python
# these parameters take ~1 minute to fit to have a nice visualisation, 
# scale down the size of the image/the patch to increase speed.
images = np.stack([make_circle(1001), make_cross(1001)])
transformer = OrientationScoreTransformer(wavelet_dim=501, num_slices=12)
transformer.fit(images)
oriented_circle, oriented_cross  = transformer.transform(images)
```

```python
f, axs = plt.subplots(ncols=4, figsize=(20, 5))
for ax in axs:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
axs[0].matshow(transformer._cake_slices[0])

# zoom a little bit on the wavelet
# zoomed_slice = slice(int(3*transformer.wavelet_dim/8),int(5*transformer.wavelet_dim/8))
zoomed_slice = slice(int(3.75*transformer.wavelet_dim/8),int(4.25*transformer.wavelet_dim/8))

axs[1].matshow(transformer._wavelets[0].imag[zoomed_slice, zoomed_slice])
axs[2].matshow(images[0])
axs[3].matshow(oriented_circle.imag[:, :, 0])
f.savefig('../report/plots/cake_wavelet.png')
```

```python
f, ax = plt.subplots(figsize=(10, 7))
ax.matshow(transformer._wavelets[0].imag[zoomed_slice, zoomed_slice], cmap='gray')
```

```python
import ipyvolume
ipyvolume.quickvolshow(150*(val>80),level=[150], level_width=[10])
```
