```python
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
%matplotlib inline
```

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from templatematching.utils import load_patches, read_norm_img, read_eye_annotations, read_pgm
```

```python
img_no = 1521
patches, labels = load_patches(img_no)
```

```python
X_train, X_test, y_train, y_test = train_test_split(patches, labels, train_size=0.8, shuffle=True, random_state=1)

```

```python
print(f'train size {X_train.shape[0]} samples')
print(f'test size {X_test.shape[0]} samples')
print(f'proportion of true labels in train set {np.round(100 * np.sum(y_train) / y_train.shape[0], 2)} %')
print(f'proportion of true labels in test set {np.round(100 * np.sum(y_test) / y_test.shape[0], 2)} %')
```

```python
def make_circle(r, x, y, true_x, true_y, ax=None):
    
    color = 'red'
    if np.sqrt((x - true_x) ** 2 + (y - true_y) ** 2) < np.sqrt(r):
        color = 'g'
    theta = np.linspace(0, 2 * np.pi, 100)
    r = np.sqrt(r)
    x1 = r*np.cos(theta) + y
    x2 = r*np.sin(theta) + x
    
    ax.plot(x2, x1, c=color)
```

```python
from templatematching.models.averager import Averager
clf = Averager()
clf.fit(X_train)
```

```python
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
image = read_norm_img(17)
img = read_pgm(17)
true_x, true_y = labels[17][0], labels[17][1]

r = 50

conv, (y, x) = clf.predict(image)

ax1.matshow(image, cmap='gray')
ax2.matshow(conv, cmap='gray')
ax3.matshow(img, cmap='gray')
make_circle(r, x, y, true_x, true_y, ax3)
ax3.scatter(true_x, true_y)
```

```python
clf.score(X_test, y_test)
```

```python
from templatematching.models import R2Ridge
clf = R2Ridge(template_shape=(51, 51), mu=0, spline_order=3)
clf.fit(X=X_train, y=y_train[:, 2].reshape(y_train.shape[0], 1))
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

```python
clf.score(X_test, y_test)
```

```python
from templatematching.models.logistic import R2LogReg
clf = R2LogReg(template_shape=(51, 51), mu=1e-4, spline_order=3, optimizer_steps=10, random_state=10)
clf.fit(X=X_train, y=y_train[:, 2].reshape(y_train.shape[0], 1))
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

```python
clf.score(X_test, y_test)
```
