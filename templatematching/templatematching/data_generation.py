import numpy as np


def make_cross(N):
    XX, YY = np.meshgrid(np.arange(N), np.arange(N))

    upper_left_diag = np.abs(XX - YY) < (N / 20)
    upper_right_diag = np.logical_and(
        np.abs(XX + YY) > 0.95 * N, np.abs(XX + YY) < 1.05 * N
    )
    return np.logical_or(upper_left_diag, upper_right_diag)


def make_circle(N):
    XX, YY = np.meshgrid(np.arange(N), np.arange(N))
    XY = np.stack([XX, YY], axis=-1)
    center = np.array([(N - 1) / 2, (N - 1) / 2])
    radius = N / 3
    val = np.linalg.norm(XY - center[np.newaxis, np.newaxis, :], axis=2)
    ret_one = 1 * np.logical_and(val < 1.1 * radius, val > 0.9 * radius)
    ret_two = 0 * np.logical_or(val > 1.1 * radius, val < 0.9 * radius)
    return ret_one + ret_two
