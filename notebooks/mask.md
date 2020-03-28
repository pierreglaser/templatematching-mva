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

## Mask creation (optional)

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

from templatematching.preprocessing import m_function
```

```python
r = 10

X = np.linspace(-r, r, 2 * r + 1)
Y = np.linspace(-r, r, 2 * r + 1)
x, y = np.meshgrid(X, Y)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15, 10))

disk_window = np.sqrt(x ** 2 + y ** 2) < r
ax1.imshow(disk_window)


n_order = 1
m_function_part = functools.partial(m_function, r=r, n_order=n_order)
# Compute normilizing constante
eta = dblquad(m_function_part, - np.inf, np.inf, -np.inf, np.inf)[0]
window = m_function(y, x, r=r, n_order=n_order) / eta
ax2.imshow(window)


n_order = 3
m_function_part = functools.partial(m_function, r=r, n_order=n_order)
# Compute normilizing constante
eta = dblquad(m_function_part, - np.inf, np.inf, -np.inf, np.inf)[0]
window_3 = m_function(y, x, r=r, n_order=n_order) / eta
ax3.imshow(window_3)


n_order = 5
m_function_part = functools.partial(m_function, r=r, n_order=n_order)
# Compute normilizing constante
eta = dblquad(m_function_part, - np.inf, np.inf, -np.inf, np.inf)[0]
window_5 = m_function(y, x, r=r, n_order=n_order) / eta
ax4.imshow(window_5)
```
