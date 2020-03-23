import numpy as np
from scipy.signal import convolve



def make_first_order_spline(i):
    # t0 is 0
    def spline_i(x):
        return 1 * np.logical_and(x >= i, x < i + 1)

    return spline_i


def make_k_th_order_spline(i, k):
    if k == 1:
        return make_first_order_spline(i)

    def spline_k_i(x):
        bk1 = make_k_th_order_spline(i, k - 1)
        bk2 = make_k_th_order_spline(i + 1, k - 1)

        ret1 = (x - i) / (i + k - 1 - i) * bk1(x)
        ret2 = (i + k - x) / (i + k - (i + 1)) * bk2(x)

        return ret1 + ret2

    return spline_k_i


def make_spline_first_derivative(i, k):
    def b_i_1_derivative(x):
        return 0 * x
    
    if k == 1:
        return b_i_1_derivative
    
    def d_b_ik(x):
        bik1 = make_k_th_order_spline(i, k-1)
        bik2 = make_k_th_order_spline(i+1, k-1)
        
        ret1 = bik1(x) / (i+k - i)
        ret2 = bik2(x) / (i+k+1 - (i+1))
        return k * (ret1 - ret2)
    return d_b_ik


def make_2d_spline_patch(n, k, l):
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    
    s_x = make_k_th_order_spline(k, n)
    s_y = make_k_th_order_spline(l, n)
    
    spline_x = s_x(x)
    spline_y = s_y(y)
    
    return np.outer(spline_y, spline_x)


def make_2d_spline(n, k, l, Nx, Ny, Nk, Nl):
    """
    Inputs:
    -------
    n (int):
        Spline degree

    Return the 2D spline B^n_{sk, sl} = B^n(x / sk) * B^n(y / sl)
    """
    x = np.arange(Nx)
    y = np.arange(Ny)

    s_x = make_k_th_order_spline(k, n)
    s_y = make_k_th_order_spline(l, n)

    s_k = Nx / Nk
    s_l = Ny / Nl

    spline_x = s_x(x / s_k)
    spline_y = s_y(y / s_l)

    return np.outer(spline_y, spline_x)

############################################   Fonctions coming from the paper    ########################################################

def spline_0(x, start=-51, stop=51, granularity=1000):
    """
    Return first order spline
    """
    return np.logical_and(-0.5 <= x, x <= 0.5).astype(int)


def make_spline_n_deg(n, start=-51, stop=51, granularity=1000):
    """
    Return a n-order spline as define in the paper B^n(x) = I_[-0.5, 0.5] (*)^ n I_[-0.5, 0.5]

    Inputs:
    -------
    n (int):
        The spline's order
    """


    if n == 0:
        return spline_0

    def spline_n(x):

        s = np.linspace(start, stop, granularity)

        # Get the index
        index = np.round(granularity * (x - start) / (stop - start) -1).astype(int)

        # Build previous spline
        spline_prev = make_spline_n_deg(n - 1)

        # Create function
        s_prev = spline_prev(s)

        # Create indicator fonction
        ind = np.logical_and(-0.5 <= s, s <= 0.5).astype(int)

        # Convolve
        conv = np.convolve(ind, s_prev, mode='same') / sum(ind)

        return conv[index]

    return spline_n
    

def make_2D_spline_deg_n(n, sk=1, sl=1, start=-51, stop=51, granularity=1000):
    """
    Inputs:
    -------
    n (int):
        Spline degree
    sk (float):
        scale factor x-axis
    sl (float):
        scale factor y-axis

    Return the 2D spline B^n_{sk, sl} = B^n(x / sk) * B^n(y / sl)
    """
    Bx = make_spline_n_deg(n, start=start, stop=stop, granularity=granularity)
    By = make_spline_n_deg(n, start=start, stop=stop, granularity=granularity)


    def spline_2D(x, y):

        return np.outer(Bx(x / sk),  By(y / sl))

    return spline_2D


