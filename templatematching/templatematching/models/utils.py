import functools
import numpy as np

from scipy.integrate import dblquad

from ..preprocessing import m_function


def make_template_mass(r=50, n_order=5):
    """
    This function returns a smooth disk-mask for template
    matching using m_function (c.f. preprocessing m_function)

    Inputs:
    -------
    radius (int):
        The radius of the disk
    n_order (int):
        The order to be used in m_function
    """
    X = np.linspace(-r, r, 2 * r + 1)
    Y = np.linspace(-r, r, 2 * r + 1)
    x, y = np.meshgrid(X, Y)

    m_function_part = functools.partial(m_function, r=r, n_order=n_order)
    # Compute normilizing constante
    eta = dblquad(m_function_part, -np.inf, np.inf, -np.inf, np.inf)[0]
    window = m_function(y, x, r=r, n_order=n_order) / eta

    return window
