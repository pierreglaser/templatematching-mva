import numpy as np



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