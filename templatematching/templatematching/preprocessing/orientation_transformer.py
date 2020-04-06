import numpy as np

from math import factorial

from numpy import pi
from numpy.fft import ifft2
from scipy.signal import fftconvolve
from joblib import Parallel, delayed

from ..spline import make_k_th_order_spline
from scipy.fft import fftn, ifftn
from scipy.signal.signaltools import _centered


def _broadcasted_convolution(in1, in2, orig_shape):
    fwr = in1[:, :, :, np.newaxis] * np.swapaxes(in2[:, :, :, np.newaxis], 0, 3)
    fwr = fwr.reshape(len(in1), *fwr.shape[1:3], len(in2))

    e = ifftn(fwr, s=fwr.shape[1:3], axes=[1, 2])
    return _centered(e, (len(in1), *orig_shape, len(in2)))


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
        ret = np.exp(-rho) * sum((rho ** k) / factorial(k) for k in range(N + 1))
        return ret

    return M_n


class OrientationScoreTransformer:
    """Transform a 2D image into a 3D representation taking in account the
    orientation of the patterns inside the image.
    """

    def __init__(
        self,
        wavelet_dim,
        num_slices,
        spline_order=3,
        mn_order=8,
        bandwidth=5,
        convolution_mode="same",
        batch_size=50,
        n_jobs=1
    ):
        self.wavelet_dim = wavelet_dim
        self.num_slices = num_slices
        self.spline_order = spline_order
        self.mn_order = mn_order
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self.convolution_mode = convolution_mode
        self.s_theta = 2 * np.pi / self.num_slices
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self._wavelets = []

        self._B_k = make_k_th_order_spline(0, self.spline_order)
        self._M_n = make_m_function_cake(self.mn_order)

        self._wavelets = []  # dim: (dimx, dimy)
        self._cake_slices = []

        for orientation in np.linspace(0, pi, self.num_slices):
            w, cake_slice = self._make_cake_wavelet(orientation=orientation)
            self._wavelets.append(w)
            self._cake_slices.append(cake_slice)

    def _legacy_transform(self, X):
        transformed_X = []

        batch_size = min(self.batch_size, X.shape[0])
        for w in self._wavelets:
            w = w.reshape(1, *w.shape)  # convolve over a full batch of images
            convolved_img = np.zeros(X.shape).astype(np.complex64)
            for i in range(int(X.shape[0] / batch_size)):
                X_batch = X[i * self.batch_size: (i + 1) * self.batch_size, :, :]
                convolved_img[
                    i * self.batch_size: (i + 1) * self.batch_size, :, :
                ] = fftconvolve(X_batch, w, mode=self.convolution_mode)
            transformed_X.append(convolved_img)

        return np.stack(transformed_X, axis=-1)

    def transform(self, X):
        wavelets = np.array(self._wavelets)
        ss = (X.shape[1] + wavelets.shape[1] - 1,
              X.shape[2] + wavelets.shape[2] - 1)
        fftws = fftn(wavelets, s=ss, axes=[1, 2])
        fftimgs = fftn(X, s=ss, axes=[1, 2])

        from math import ceil
        batched_convs = Parallel(prefer="processes", n_jobs=self.n_jobs)(
            delayed(_broadcasted_convolution)(
                fftimgs[i * self.batch_size: (i + 1) * self.batch_size, :, :],
                fftws, orig_shape=X.shape[1:])
            for i in range(ceil(X.shape[0] / self.batch_size)))
        return np.concatenate(batched_convs)

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def _make_cake_wavelet(self, orientation):
        gaussian_window = _make_gaussian_patch(
            N=self.wavelet_dim, sigma=self.wavelet_dim / 4
        )
        cake_slice = self._make_cake_slice(orientation=orientation)
        w = ifft2(np.fft.ifftshift(cake_slice)) * gaussian_window

        # account for circular boundary conditions of fourier constructs.
        w = self._rearrange_wavelet(w)
        return w, cake_slice

    def _make_cake_slice(self, orientation):
        rhos, phis = make_polar_coordinates(self.wavelet_dim, self.bandwidth)
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
