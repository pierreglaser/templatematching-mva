import numpy as np

from scipy.signal import fftconvolve, correlate

from ..spline import (
    discrete_spline_2D,
    discrete_spline_3D,
    discrete_spline,
    discrete_spline_second_derivative,
)

from .base import (
    PatchRegressorBase,
    SplineRegressorBase,
    TemplateCrossCorellatorBase,
)
from ..preprocessing import OrientationScoreTransformer


class R2Ridge(SplineRegressorBase, PatchRegressorBase):
    def __init__(
        self,
        template_shape,
        splines_per_axis,
        spline_order=2,
        mu=0,
        lbd=0,
        verbose=0,
        solver="dual",
        random_state=None,
        eye="left",
    ):
        assert template_shape[0] == template_shape[1]
        assert solver in ["primal", "dual"], "solver must be primal or dual"
        SplineRegressorBase.__init__(
            self, template_shape, splines_per_axis, spline_order=spline_order
        )
        PatchRegressorBase.__init__(
            self, patch_shape=template_shape, eye=eye, random_state=random_state,
        )

        self.model_name = "Linear Ridge"
        self.mu = mu
        self.lbd = lbd
        self.verbose = verbose

        self.solver = solver
        self._is_fitted = False
        self._cached_template = None

    @TemplateCrossCorellatorBase.template.getter
    def template(self):
        if self._is_fitted:
            if self._template is not None:
                return self._template
            else:
                return self._reconstruct_template()
        else:
            raise AttributeError("No template yet: Classifier not fitted")

    def _fit_patches(self, X, y):
        self._check_params(X)
        S = self._create_s_matrix(X)
        R = self._create_R_matrix()
        if self.solver == "primal":
            c = np.linalg.lstsq(
                S.T @ S + self.lbd * R + self.mu * np.eye(S.shape[1]),
                S.T @ y,
                rcond=None,
            )[0]
        elif self.solver == "dual":
            if self.lbd == 0:
                c = S.T @ np.linalg.inv(S @ S.T + self.mu * np.eye(S.shape[0])) @ y
            else:
                B = self.mu * np.eye(S.shape[1]) + self.lbd * R
                B_inv = np.linalg.inv(B)
                c = (
                    B_inv
                    @ S.T
                    @ np.linalg.inv(S @ B_inv @ S.T + np.eye(S.shape[0]))
                    @ y
                )
        else:
            raise ValueError(f"solver must be 'primal' or 'dual', got '{self.solver}'")
        self._S, self._spline_coef = S, c

    def _reconstruct_template(self):
        _, Nx, Ny, Nk, Nl, sk, sl = self._get_dims()
        B = self._make_unit_spline(sk, sl)

        impulses = np.zeros((Nx, Ny))
        impulses[::sk, ::sl] = self._spline_coef.reshape(Nk, Nl)

        self._template = fftconvolve(impulses, B, mode="same")
        return self._template

    def _create_s_matrix(self, X):
        num_samples, Nx, Ny, Nk, Nl, sk, sl = self._get_dims()
        S = np.zeros((num_samples, Nk * Nl))
        B = self._make_unit_spline(sk, sl)
        B = B.reshape(1, *B.shape)

        convolved_X = fftconvolve(X, B, mode="same")
        S = convolved_X[:, ::sk, ::sl].reshape(num_samples, Nk * Nl)
        S /= np.linalg.norm(S, axis=0, keepdims=True)

        return S

    def _create_R_matrix(self):
        _, _, _, Nk, Nl, sk, sl = self._get_dims()
        x = np.linspace(-int(Nk / 2), int(Nk / 2), Nk)
        y = np.linspace(-int(Nl / 2), int(Nl / 2), Nl)

        xs = np.array([[xi - xk for xi in y] for xk in x])
        ys = np.array([[yi - yk for yi in y] for yk in y])

        Bxk = -1 / sk * discrete_spline_second_derivative(xs, 2 * self.spline_order + 1)
        Bxl = sl * discrete_spline(xs, 2 * self.spline_order + 1)
        Byk = sk * discrete_spline(ys, 2 * self.spline_order + 1)
        Byl = -1 / sl * discrete_spline_second_derivative(ys, 2 * self.spline_order + 1)

        R = np.kron(Bxk, Bxl) + np.kron(Byk, Byl)

        return R

    def _make_unit_spline(self, sk, sl):
        # Our spline is always defined on [-2.5, 2.5] (may be a problem if we
        # change the order of the spline, as B_k is defined on [-k/2, k/2]) the
        # granularity of the grid impacts the percieved width of the spline.
        x_grid = np.array(range(-int(5 * sk / 2), int(5 * sk / 2) + 1)) / sk
        y_grid = np.array(range(-int(5 * sl / 2), int(5 * sl / 2) + 1)) / sl
        B = discrete_spline_2D(x_grid, y_grid, self.spline_order)
        return B

    def _check_params(self, X):
        _, Nx, Ny = X.shape  # Nx, Ny: training images shape
        Nk, Nl = self.splines_per_axis
        # The convention is that images are "centered" on the origin:
        # A valid image should thus have 2n + 1 pixels.
        assert ((Nx - 1) % (Nk - 1)) == 0
        assert ((Ny - 1) % (Nl - 1)) == 0
        self._X_shape = X.shape

    def _get_dims(self):
        num_samples, Nx, Ny = self._X_shape
        Nk, Nl = self.splines_per_axis
        sk, sl = (Nx - 1) // (Nk - 1), (Ny - 1) // (Nl - 1)  # fmt: off
        return num_samples, Nx, Ny, Nk, Nl, sk, sl


class SE2Ridge(SplineRegressorBase, PatchRegressorBase):
    def __init__(
        self,
        template_shape,
        splines_per_axis,
        wavelet_dim,
        num_orientation_slices=12,
        spline_order=2,
        mu=0,
        verbose=0,
        solver="dual",
    ):
        assert template_shape[0] == template_shape[1]
        assert solver in ["primal", "dual"], "solver must be primal or dual"

        SplineRegressorBase.__init__(
            self, template_shape, splines_per_axis, spline_order=spline_order
        )
        PatchRegressorBase.__init__(self, patch_shape=template_shape)

        self.name = "SE2 Ridge"
        self.mu = mu
        self.verbose = verbose

        self.solver = solver
        self._is_fitted = False
        self._template = None
        self._ost = OrientationScoreTransformer(
            wavelet_dim=wavelet_dim, num_slices=num_orientation_slices
        )

    @TemplateCrossCorellatorBase.template.getter
    def template(self):
        if self._is_fitted:
            if self._template is not None:
                return self._template
            else:
                return self._reconstruct_template()
        else:
            raise AttributeError("No template yet: Classifier not fitted")

    def predict(self, X):
        # TODO: put this method in a Mixin Class.
        X = self._ost.transform(X).imag
        template = self.template.reshape(1, *self.template.shape)
        convs = correlate(X, template, mode="same", method="fft")
        positions = []
        for i in range(len(X)):
            (y, x, _) = np.unravel_index(np.argmax(convs[i]), convs[i].shape)
            positions.append([x, y])
        return convs, np.array(positions)
        return TemplateCrossCorellatorBase.predict(self, X)

    def _fit_patches(self, X, y):
        X = self._ost.fit_transform(X).imag  # can also take the modulus
        self._check_params(X)
        S = self._create_s_matrix(X)
        if self.solver == "primal":
            c = np.linalg.lstsq(
                S.T @ S + self.mu * np.eye(S.shape[1]), S.T @ y, rcond=None,
            )[0]
        elif self.solver == "dual":
            c = S.T @ np.linalg.inv(S @ S.T + self.mu * np.eye(S.shape[0])) @ y
        else:
            raise ValueError(f"solver must be 'primal' or 'dual', got '{self.solver}'")
        self._S, self._spline_coef = S, c

    def _create_s_matrix(self, X):
        num_samples, Nx, Ny, Nt, Nk, Nl, Nm, sk, sl, sm = self._get_dims()
        S = np.zeros((num_samples, Nk * Nl * Nm))
        B = self._make_unit_spline(sk, sl, sm)
        B = B.reshape(1, *B.shape)

        convolved_X = fftconvolve(X, B, mode="same")
        S = convolved_X[:, ::sk, ::sl, ::sm].reshape(num_samples, Nk * Nl * Nm)
        S /= np.linalg.norm(S, axis=0, keepdims=True)
        return S

    def _reconstruct_template(self):
        _, Nx, Ny, Nt, Nk, Nl, Nm, sk, sl, sm = self._get_dims()
        B = self._make_unit_spline(sk, sl, sm)

        impulses = np.zeros((Nx, Ny, Nt))
        impulses[::sk, ::sl, ::sm] = self._spline_coef.reshape(Nk, Nl, Nm)

        self._template = fftconvolve(impulses, B, mode="same")
        return self._template

    def _make_unit_spline(self, sk, sl, st):
        # Our spline is always defined on [-2.5, 2.5] (may be a problem if we
        # change the order of the spline, as B_k is defined on [-k/2, k/2]) the
        # granularity of the grid impacts the percieved width of the spline.

        x_grid = np.array(range(-int(5 * sk / 2), int(5 * sk / 2) + 1)) / sk
        y_grid = np.array(range(-int(5 * sl / 2), int(5 * sl / 2) + 1)) / sl
        t_grid = np.array(range(-int(5 * st / 2), int(5 * st / 2) + 1)) / st

        B = discrete_spline_3D(x_grid, y_grid, t_grid, self.spline_order)
        return B

    def _check_params(self, X):
        _, Nx, Ny, Nt = X.shape  # Nx, Ny, Nt: training images shape
        Nk, Nl, Nm = self.splines_per_axis  # fmt: off
        assert (Nx - 1) % (Nk - 1) == 0
        assert (Ny - 1) % (Nl - 1) == 0
        assert Nt % Nm == 0  # theta is not centered
        self._X_shape = X.shape

    def _get_dims(self):
        num_samples, Nx, Ny, Nt = self._X_shape
        Nk, Nl, Nm = self.splines_per_axis  # fmt: off
        sk, sl = (Nx - 1) // (Nk - 1), (Ny - 1) // (Nl - 1)
        sm = Nt // Nm
        return num_samples, Nx, Ny, Nt, Nk, Nl, Nm, sk, sl, sm
