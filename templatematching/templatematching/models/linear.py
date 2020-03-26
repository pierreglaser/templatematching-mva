import numpy as np

from itertools import product

from scipy.signal import convolve2d, correlate2d

from ..spline import spline_kl_to_xy, discrete_spline_2D

from .utils import make_template_mass


class R2Ridge(object):
    def __init__(self, template_shape, spline_order=2, mu=0, verbose=0):
        self.model_name = 'Linear Ridge'
        self.template_shape = template_shape
        self.spline_order = spline_order
        self.mu = mu
        self.verbose = verbose
        self._S = None
        self._template = None
        self._template_full = None
        self._mask = None

    def fit(self, X, y):
        S = self._make_s_matrix(X)
        c = np.linalg.lstsq(
            S.T @ S + self.mu * np.eye(S.shape[1]), S.T @ y, rcond=None
        )
        # record fitting information (input shape, S matrix, coefficients etc.)
        self._S, self._Nx, self._Ny = S, X.shape[1], X.shape[2]
        self.spline_coef = c[0]
        self._mask = make_template_mass(int(X.shape[1]/2))

    def predict(self, X):

        conv = correlate2d(X, self._template, mode="same")
        (y, x) = np.where(conv == np.amax(conv))

        return conv, (y, x)

    def score(self, X, y, radius_criteria=50):

        num_sample = X.shape[0]
        score = 0

        for i in range(num_sample):
            image = X[i]
            _, (pred_y, pred_x) = self.predict(image)
            true_x, true_y = y[i][0], y[i][1]

            if np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2) < np.sqrt(radius_criteria):
                score += 1

        total_score = np.round(score / num_sample * 100, 2)
        print(f'Score was computed on {num_sample} samples: \n')
        print(f'Model {self.model_name} accuracy: {total_score} %')


    def reconstruct_template(self):
        final_template = np.zeros((self._Nx, self._Ny))
        _x_idxs, _y_idxs = self._make_subsampled_grid(
            self._Nx, self._Ny, *self.template_shape, centered=False
        )
        spline_width = (self._B.shape[0] - 1) // 2

        for i, (x, y) in enumerate(product(_x_idxs, _y_idxs)):
            _sx = slice(x - spline_width, x + spline_width + 1)
            _sy = slice(y - spline_width, y + spline_width + 1)
            if _sx.start < 0 or _sy.start < 0:
                continue

            if _sx.stop > self._Nx or _sy.stop > self._Ny:
                continue

            final_template[_sx, _sy] += self.spline_coef[i] * self._B

        self._template_full = final_template
        self._template = self._mask * self._template_full

        return final_template * self._mask

    def _make_s_matrix(self, X):
        num_samples, Nx, Ny = X.shape  # Nx, Ny: training images shape
        Nk, Nl = self.template_shape  # Nk, Nl: number of splines in each axis
        sk, sl = Nx / Nk, Ny / Nl

        S = np.zeros((num_samples, Nk * Nl))
        self._B = self._make_unit_spline(sk, sl)

        # centers of each 2D spline used to create the template, compressed
        # as a carthesian product
        _x_idxs, _y_idxs = self._make_subsampled_grid(
            Nx, Ny, Nk, Nl, centered=False
        )

        for i in range(num_samples):
            if self.verbose and i % 100 == 0:
                print("Creating S matrix - Patch {}".format(i))

            convolved_sample_i = convolve2d(X[i], self._B, mode="same")

            # Get the corresponding points in(x, y) grid and flatten
            S[i] = convolved_sample_i[_x_idxs, :][:, _y_idxs].flatten()

        S /= np.linalg.norm(S, axis=0)[np.newaxis, :]

        return S

    def _make_subsampled_grid(self, Nx, Ny, Nk, Nl, centered=True):
        # generate a subsampled grid between:
        # - [-Nx/2, Nx/2] x [Ny/2,Ny/2] if centered
        # - [0, Nx] x [0, Ny] if not centered
        # containing Nk points in the x-axis, and Nl points in the y-axis
        sk, sl = Nx / Nk, Ny / Nl

        k_grid, l_grid = np.arange(Nk), np.arange(Nl)
        subsampled_x_grid, subsampled_y_grid = spline_kl_to_xy(
            k_grid, l_grid, sk, sl
        )

        if centered:
            subsampled_x_grid -= (Nx - 1) // 2
            subsampled_y_grid -= (Ny - 1) // 2
        else:
            assert subsampled_x_grid[0] == 0
            assert subsampled_y_grid[0] == 0

        return subsampled_x_grid, subsampled_y_grid

    def _make_unit_spline(self, sk, sl):
        # Our spline is always defined on [-2.5, 2.5] (may be a problem if we
        # change the order of the spline, as B_k is defined on [-k/2, k/2]) the
        # granularity of the grid impacts the percieved width of the spline.

        x_grid = np.array(range(-int(5 * sk / 2), int(5 * sk / 2) + 1)) / sk
        y_grid = np.array(range(-int(5 * sl / 2), int(5 * sl / 2) + 1)) / sl

        B = discrete_spline_2D(x_grid, y_grid, self.spline_order)
        return B
