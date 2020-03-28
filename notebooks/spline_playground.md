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
%matplotlib inline
%load_ext autoreload
%autoreload 2
from templatematching.utils import read_pgm, read_eye_annotations
import numpy as np
```

```python
image = read_pgm(0)
eyes = read_eye_annotations(image_nos=3)
```

```python
f, ax = plt.subplots()
ax.matshow(image, cmap='gray')
_ = ax.scatter([left_eye[0]], [left_eye[1]], c='green')
_ = ax.scatter([right_eye[0]], [right_eye[1]], c='green')
```

### Splines

```python
from templatematching.spline import (
    make_k_th_order_spline,
    make_spline_first_derivative,
    make_2d_spline_patch,
    make_2D_spline_deg_n,
    make_spline_n_deg
)
```

```python
import itertools
plt.subplots(nrows=10, ncols=5, sharex=True, sharey=True, figsize=(16, 8))
j = 0
for i, k in itertools.product(range(10), range(1, 6)):
    Bik = make_k_th_order_spline(i, k)
    xs = np.linspace(0, 10, 100)
    ys = Bik(xs)
    
    j+=1
    plt.subplot(10, 5, j)
    plt.plot(xs, ys)
```

```python
import itertools
plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True ,figsize=(16, 8))
j = 0
for i, k in itertools.product(range(3), range(1, 6)):
    Bik = make_k_th_order_spline(i, k)
    dBik = make_spline_first_derivative(i, k)
    xs = np.linspace(0, 10, 100)
    
    ys = Bik(xs)
    ys_prime = dBik(xs)
    
    j+=1
    plt.subplot(3, 5, j)
    plt.plot(xs, ys, c='b')
    plt.plot(xs, ys_prime, c='r')
```

```python
f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)

p = make_2d_spline_patch(1, 9, 5)
_ = ax1.matshow(p)

p = make_2d_spline_patch(3, 2, 3)
_ = ax2.matshow(p)
```

```python
import itertools
plt.subplots(nrows=1, ncols=6, sharex=True, sharey=True, figsize=(25, 35))
j = 0
for k in range(0, 6):
    Bik = make_spline_n_deg(k)
    xs = np.linspace(-10, 10, 100)
    ys = Bik(xs)
    
    j+=1
    plt.subplot(10, 6, j)
    plt.plot(xs, ys)
```

```python
B2d = make_2D_spline_deg_n(2)
```

```python
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 35))

Nx = Ny = 101
Nk = Nl = 51

sk = (Nx - 1) / (Nk - 1)
sl = (Ny - 1) / (Nk - 1)

x = np.linspace(-int(5 * sk / 2) + 1, int(5 * sk / 2), 10 * int(sk / 2))
y = np.linspace(-int(5 * sl / 2) + 1, int(5 * sk / 2), 10 * int(sk / 2))

B2d = make_2D_spline_deg_n(0, sk=sk, sl=sl)
ax1.imshow(B2d(x, y))

B2d = make_2D_spline_deg_n(1, sk=sk, sl=sl)
ax2.imshow(B2d(x, y))

B2d = make_2D_spline_deg_n(2, sk=sk, sl=sl)
ax3.imshow(B2d(x, y))

B2d = make_2D_spline_deg_n(3, sk=sk, sl=sl)
ax4.imshow(B2d(x, y))
```

## Discrete splines

```python
from templatematching.spline import (
    discrete_spline,
    discrete_spline_first_derivative,
    discrete_spline_second_derivative,
    discrete_spline_2D)
```

```python
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 35))

Nx = Ny = 101
Nk = Nl = 51

sk = Nx / Nk
sl = Ny / Nk

x = np.linspace(-int( 5 * sk /2), int( 5 * sk /2), 5 * sk ) / sk
y = np.linspace(-int( 5 * sl /2), int( 5 * sl /2), 5 * sl ) / sl

B2d = discrete_spline_2D(x, y, 9)
ax1.imshow(B2d)

B2d =discrete_spline_2D(x, y, 1)
ax2.imshow(B2d)

B2d = discrete_spline_2D(x, y, 2)
ax3.imshow(B2d)

B2d = discrete_spline_2D(x, y, 3)
ax4.imshow(B2d)

```

```python
import itertools
plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(15, 15))
j = 0
for i in range(1, 6):
    xs = np.linspace(-10, 10, 100)
    
    ys = discrete_spline(xs, i)
    zs = discrete_spline_first_derivative(xs, i)
    ts = discrete_spline_second_derivative(xs, i)

    j+=1
    plt.subplot(10, 5, j)
    plt.plot(xs, ys, c='b')
    plt.plot(xs, zs, c='r')
    plt.plot(xs, ts, c='g')
```

```python
def _make_R_matrix(Nk, Nl, sk, sl, n):
    x = np.linspace(-int( Nk /2), int( Nk /2), Nk)
    y = np.linspace(-int( Nl /2), int( Nl /2), Nl)
    
    xs = np.array([ [xi - xk for xi in y] for xk in x])
    ys = np.array([ [yi - yk for yi in y] for yk in y])
    
    Bxk = - 1 / sk * discrete_spline_second_derivative(xs, 2 * n +1)
    Bxl = sl * discrete_spline(xs, 2 * n + 1)
    Byk = sk * discrete_spline(ys, 2 * n +1)
    Byl = - 1 / sl * discrete_spline_second_derivative(ys, 2 * n + 1)
    
    return np.kron(Bxk, Bxl) + np.kron(Byk, Byl)
    
    
```

```python
R = _make_R_matrix(Nk, Nl, sk, sl, 3)
plt.matshow(R)
R.shape
```

```python
51*51
```

```python

```
