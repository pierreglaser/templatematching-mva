import numpy as np

from scipy.signal import fftconvolve

from ..spline import discrete_spline_2D, discrete_spline_3D
from .base import (
    PatchRegressorBase,
    SplineRegressorBase,
    TemplateCrossCorellatorBase,
)
from .utils import make_template_mass


class R2Ridge(SplineRegressorBase, PatchRegressorBase):
    def __init__(
        self,
        template_shape,
        splines_per_axis,
        spline_order=2,
        mu=0,
        verbose=0,
        solver="dual",
        random_state=None,
    ):
        assert template_shape[0] == template_shape[1]
        assert solver in ["primal", "dual"], "solver must be primal or dual"
        SplineRegressorBase.__init__(
            self, template_shape, splines_per_axis, spline_order=spline_order
        )
        PatchRegressorBase.__init__(
            self, patch_shape=template_shape, random_state=random_state
        )

        self.model_name = "Linear Ridge"
        self.mu = mu
        self.verbose = verbose
        self._mask = make_template_mass(int(template_shape[0] / 2))

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
        if self.solver == "primal":
            c = np.linalg.lstsq(
                S.T @ S + self.mu * np.eye(S.shape[1]), S.T @ y, rcond=None,
            )[0]
        elif self.solver == "dual":
            c = S.T @ np.linalg.inv(S @ S.T + self.mu * np.eye(S.shape[0])) @ y
        else:
            raise ValueError(
                f"solver must be 'primal' or 'dual', got '{self.solver}'"
            )
        self._S, self._spline_coef = S, c

    def _reconstruct_template(self):
        _, Nx, Ny, Nk, Nl, sk, sl = self._get_dims()
        B = self._make_unit_spline(sk, sl)

        impulses = np.zeros((Nx, Ny))
        impulses[::sk, ::sl] = self._spline_coef.reshape(Nk, Nl)

        self._template = fftconvolve(impulses, B, mode="same")
        self._masked_template = self._mask * self._template
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

        super().__init__(splines_per_axis, spline_order)
        self.name = "SE2 Ridge"
        self.mu = mu
        self.verbose = verbose

        self.solver = solver
        self._is_fitted = False
        self._template = None

    def _fit_patches(self, X, y):
        self._check_params(self, X)
        S = self._create_s_matrix(X)
        if self.solver == "primal":
            c = np.linalg.lstsq(
                S.T @ S + self.mu * np.eye(S.shape[1]), S.T @ y, rcond=None,
            )[0]
        elif self.solver == "dual":
            c = S.T @ np.linalg.inv(S @ S.T + self.mu * np.eye(S.shape[0])) @ y
        else:
            raise ValueError(
                f"solver must be 'primal' or 'dual', got '{self.solver}'"
            )
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
        impulses[::sk, ::sl, ::Nt] = self._spline_coef.reshape(Nk, Nl, Nt)

        self._template = fftconvolve(impulses, B)
        self._masked_template = self._mask * self._template_full
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
        assert (Nx % Nk) == 0
        assert (Ny % Nl) == 0
        assert (Nt % Nm) == 0
        self._X_shape = X.shape

    def _get_dims(self):
        num_samples, Nx, Ny, Nt = self._X_shape
        Nk, Nl, Nm = self.splines_per_axis  # fmt: off
        sk, sl, sm = Nx // Nk, Ny // Nl, Nt // Nm  # fmt: off
        return num_samples, Nx, Ny, Nt, Nk, Nl, Nm, sk, sl, sm
