import numpy as np

from math import factorial

from numpy import pi
from numpy.fft import ifft2
from scipy.signal import convolve2d

from .spline import make_k_th_order_spline


def _make_gaussian_patch(N, sigma):
    XX, YY = np.meshgrid(np.arange(N), np.arange(N))
    XY = np.stack([XX, YY], axis=-1)

    center = np.array([(N - 1) / 2, (N - 1) / 2])
    XY_centered = XY - center[np.newaxis, np.newaxis, :]

    gaussian_window = np.exp(np.linalg.norm(XY_centered / sigma, axis=2) ** 2)
    return gaussian_window


def make_polar_coordinates(N, bandwidth=5):
    XX, YY = np.meshgrid(
        bandwidth / N * (np.arange(N) - (N - 1) / 2),
        bandwidth / N * (np.arange(N) - (N - 1) / 2),
    )
    XY = XX + 1j * YY
    return np.abs(XY), np.mod(np.angle(XY) + pi, 2 * pi)


def make_m_function_cake(N):
    def M_n(rho, t=0.5):
        rho = rho ** 2 / t
        ret = np.exp(-rho) * sum(
                (rho ** k) / factorial(k) for k in range(N + 1)
                )
        return ret

    return M_n


class OrientationScoreTransformer:
    """Transform a 2D image into a 3D representation taking in account the
    orientation of the patterns inside the image.
    """

    def __init__(
        self,
        patch_size,
        num_slices,
        spline_order=3,
        mn_order=8,
        bandwidth=5,
        convolution_mode="full",
    ):
        self.patch_size = patch_size
        self.num_slices = num_slices
        self.spline_order = spline_order
        self.mn_order = mn_order
        self.bandwidth = bandwidth
        self.convolution_mode = convolution_mode
        self.s_theta = 2 * np.pi / self.num_slices

    def fit(self, X, y=None):
        self._wavelets = []

        self._B_k = make_k_th_order_spline(0, self.spline_order)
        self._M_n = make_m_function_cake(self.mn_order)

        self._wavelets = []
        self._cake_slices = []

        for orientation in np.linspace(0, pi, self.num_slices):
            w, cake_slice = self._make_cake_wavelet(orientation=orientation)
            self._wavelets.append(w)
            self._cake_slices.append(cake_slice)

    def transform(self, X):
        transformed_X = []
        for w in self._wavelets:
            convolved_img = convolve2d(X, w.imag, mode=self.convolution_mode)
            transformed_X.append(convolved_img)
        return np.stack(transformed_X, axis=-1)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def _make_cake_wavelet(self, orientation):
        gaussian_window = _make_gaussian_patch(
            N=self.patch_size, sigma=self.patch_size / 4
        )
        cake_slice = self._make_cake_slice(orientation=orientation)
        w = ifft2(np.fft.ifftshift(cake_slice)) * gaussian_window

        # account for circular boundary conditions of fourier constructs.
        w = self._rearrange_wavelet(w)
        return w, cake_slice

    def _make_cake_slice(self, orientation):
        rhos, phis = make_polar_coordinates(self.patch_size, self.bandwidth)
        # the + (spline_order/2) is necessary to recenter my spline orientation
        return self._B_k(
            (np.mod(phis - orientation, 2 * pi) - pi / 2) / self.s_theta
            + (self.spline_order) / 2
        ) * self._M_n(rhos)

    def _rearrange_wavelet(self, w):
        for l in range(w.shape[0]):
            w[l] = np.roll(w[l, :], w.shape[0] // 2)

        for c in range(w.shape[1]):
            w[:, c] = np.roll(w[:, c], w.shape[1] // 2)
        return w
