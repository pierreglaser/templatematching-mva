import numpy as np

from math import ceil

from scipy import sparse
from scipy.sparse import csr_matrix, kron
from scipy.sparse.linalg import lsqr as sparse_lsqr
from scipy.sparse.linalg import inv as sparse_inv

from scipy.signal import fftconvolve, correlate

from scipy.integrate import quad

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
        batch_size=50,
        mu=0,
        lbd=0,
        verbose=0,
        solver="dual",
        random_state=None,
        eye="left",
        n_jobs=1,
    ):
        assert template_shape[0] == template_shape[1]
        assert solver in ["primal", "dual"], "solver must be primal or dual"
        SplineRegressorBase.__init__(
            self, template_shape, splines_per_axis, spline_order=spline_order
        )
        PatchRegressorBase.__init__(
            self,
            patch_shape=template_shape,
            eye=eye,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.model_name = "Linear Ridge"
        self.mu = mu
        self.lbd = lbd
        self.verbose = verbose
        self.batch_size = batch_size
        self.solver = solver
        self._is_fitted = False
        self._cached_template = None
        self.n_jobs = n_jobs

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
        num_samples, _, _, _, _, _, _ = self._get_dims()
        S = self._create_s_matrix(X)
        R = self._create_r_matrix()
        if self.solver == "primal":
            c = np.linalg.lstsq(
                S.T @ S + num_samples * (self.lbd * R + self.mu * np.eye(S.shape[1])),
                S.T @ y,
                rcond=None,
            )[0]
        elif self.solver == "dual":
            if self.lbd == 0:
                c = (
                    S.T
                    @ np.linalg.inv(
                        S @ S.T + num_samples * self.mu * np.eye(S.shape[0])
                    )
                    @ y
                )
            else:
                B = num_samples * (self.mu * np.eye(S.shape[1]) + self.lbd * R)
                B_inv = sparse_inv(B)
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

        batch_size = min(self.batch_size, num_samples)

        for i in range(ceil(num_samples / batch_size)):
            X_batch = X[i * batch_size : (i + 1) * batch_size, :, :]
            convolved_X = fftconvolve(X_batch, B, mode="same")
            S[i * batch_size : (i + 1) * batch_size, :] = convolved_X[
                :, ::sk, ::sl
            ].reshape(len(X_batch), Nk * Nl)

        S /= np.linalg.norm(S, axis=0, keepdims=True)

        return S

    def _create_r_matrix(self):
        _, _, _, Nk, Nl, sk, sl = self._get_dims()
        k = np.linspace(-int(Nk / 2), int(Nk / 2), Nk)
        l = np.linspace(-int(Nl / 2), int(Nl / 2), Nl)

        ks = np.array([[ki - kk for ki in k] for kk in k])
        ls = np.array([[li - lk for li in l] for lk in l])

        Bxk = csr_matrix(-1 / sk * discrete_spline_second_derivative(ks, 2 * self.spline_order + 1))
        Bxl = csr_matrix(sl * discrete_spline(ls, 2 * self.spline_order + 1))
        Byk = csr_matrix(sk * discrete_spline(ks, 2 * self.spline_order + 1))
        Byl = csr_matrix(-1 / sl * discrete_spline_second_derivative(ls, 2 * self.spline_order + 1))

        R = kron(Bxk, Bxl) + kron(Byk, Byl)

        return R.toarray()

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
        batch_size=10,
        mu=0,
        lbd=0,
        Dxi=0,
        Deta=0,
        Dtheta=0,
        verbose=0,
        solver="dual",
        random_state=None,
        eye="left",
        n_jobs=1
    ):
        assert template_shape[0] == template_shape[1]
        assert solver in ["primal", "dual"], "solver must be primal or dual"

        SplineRegressorBase.__init__(
            self, template_shape, splines_per_axis, spline_order=spline_order
        )
        PatchRegressorBase.__init__(
            self,
            patch_shape=template_shape,
            eye=eye,
            random_state=random_state,
        )

        self.name = "SE2 Ridge"
        self.mu = mu
        self.lbd = lbd
        self.Dxi = Dxi
        self.Deta = Deta
        self.Dtheta = Dtheta
        self.verbose = verbose
        self.batch_size = batch_size
        self.solver = solver
        self._is_fitted = False
        self._template = None
        self._ost = OrientationScoreTransformer(
            wavelet_dim=wavelet_dim,
            num_slices=num_orientation_slices,
            batch_size=batch_size, n_jobs=n_jobs
        )
        self.n_jobs = n_jobs

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
        X = abs(self._ost.transform(X)) # can take imag
        template = self.template.reshape(1, *self.template.shape)
        batch_size = min(self.batch_size, X.shape[0])

        convs = np.zeros(X.shape)

        for i in range(int(X.shape[0] / batch_size)):
            X_batch = X[i * batch_size : (i + 1) * batch_size, :, :]
            convs[i * batch_size : (i + 1) * batch_size, :, :] = correlate(
                X_batch, template, mode="same", method="fft"
            )
            positions = []

        for i in range(len(X)):
            (y, x, _) = np.unravel_index(np.argmax(convs[i]), convs[i].shape)
            positions.append([x, y])
        return convs, np.array(positions)

    def _fit_patches(self, X, y):
        X = abs(self._ost.fit_transform(X))  # can also take the imag
        self._check_params(X)
        num_samples, _, _, _, _, _, _, _, _, _ = self._get_dims()
        S = self._create_s_matrix(X)
        R = self._create_r_matrix()
        R = sparse.csc_matrix(R)
        if self.solver == "primal":
            c = sparse_lsqr(
                S.T @ S + num_samples * (self.lbd * R +
                                         self.mu * sparse.eye(S.shape[1])),
                S.T @ y
            )[0]
        elif self.solver == "dual":
            if self.lbd == 0:
                c = (
                    S.T
                    @ sparse_inv(
                        S @ S.T + num_samples * self.mu * sparse.eye(S.shape[0])
                    )
                    @ y
                )
            else:
                B = num_samples * (self.mu *
                                   sparse.eye(S.shape[1]) + self.lbd * R)
                B_inv = sparse_inv(B)
                c = (
                    B_inv
                    @ S.T
                    @ sparse_inv(S @ B_inv @ S.T + sparse.eye(S.shape[0]))
                    @ y
                )
        else:
            raise ValueError(f"solver must be 'primal' or 'dual', got '{self.solver}'")
        c /= np.linalg.norm(c)
        self._S, self._spline_coef = S, c

    def _create_s_matrix(self, X):
        num_samples, Nx, Ny, Nt, Nk, Nl, Nm, sk, sl, sm = self._get_dims()
        S = np.zeros((num_samples, Nk * Nl * Nm))
        B = self._make_unit_spline(sk, sl, sm)
        B = B.reshape(1, *B.shape)

        batch_size = min(self.batch_size, X.shape[0])

        for i in range(ceil(num_samples / batch_size)):
            X_batch = X[i * batch_size : (i + 1) * batch_size, :, :]
            convolved_X = fftconvolve(X_batch, B, mode="same")
            S[i * batch_size : (i + 1) * batch_size, :] = convolved_X[
                :, ::sk, ::sl, ::sm
            ].reshape(len(X_batch), Nk * Nl * Nm)

        S /= np.linalg.norm(S, axis=0, keepdims=True)
        return S

    def _create_r_matrix(self):
        """
        Create and returns R = Dxi * Rxi + Deta * Reta + Dtheta * Rtheta

        Inputs:
        -------
        Dxi, Deta, Dtheta (float):
        """
        _, _, _, _, Nk, Nl, Nm, sk, sl, sm = self._get_dims()
        k = np.linspace(-int(Nk / 2), int(Nk / 2), Nk)
        l = np.linspace(-int(Nl / 2), int(Nl / 2), Nl)
        m = np.linspace(-int(Nm / 2), int(Nm / 2), Nm)

        ks = np.array([[ki - kk for ki in k] for kk in k])
        ls = np.array([[li - lk for li in l] for lk in l])
        ms = np.array([[mi - mk for mi in m] for mk in m])

        RIx = csr_matrix(-1 / sk * discrete_spline_second_derivative(ks, 2 * self.spline_order + 1))
        RIy = csr_matrix(sl * discrete_spline(ls, 2 * self.spline_order + 1))
        cos2_spline = lambda theta, m1, m2: np.cos(theta) ** 2 * self._util_spline(
            theta, m1, m2
        )
        RItheta = csr_matrix(
            [[quad(cos2_spline, 0, np.pi, args=(m1, m2))[0] for m1 in m] for m2 in m]
        )

        RIIx = csr_matrix(discrete_spline_second_derivative(ks, 2 * self.spline_order + 1))
        RIIy = csr_matrix(-discrete_spline_second_derivative(ls, 2 * self.spline_order + 1))
        sincos_spline = (
            lambda theta, m1, m2: np.cos(theta)
            * np.sin(theta)
            * self._util_spline(theta, m1, m2)
        )
        RIItheta = csr_matrix(
            [[quad(sincos_spline, 0, np.pi, args=(m1, m2))[0] for m1 in m] for m2 in m]
        )

        RIIIx = -RIIx.copy()
        RIIIy = -RIIy.copy()
        RIIItheta = RIItheta.copy()

        RIVx = csr_matrix(sk * discrete_spline(ks, 2 * self.spline_order + 1))
        RIVy = csr_matrix((
            -1 / sl * discrete_spline_second_derivative(ls, 2 * self.spline_order + 1)
        ))
        sin2_spline = lambda theta, m1, m2: np.sin(theta) ** 2 * self._util_spline(
            theta, m1, m2
        )
        RIVtheta = csr_matrix(
            [[quad(sin2_spline, 0, np.pi, args=(m1, m2))[0] for m1 in m] for m2 in m]
        )

        Rxtheta = csr_matrix(sk * discrete_spline(ks, 2 * self.spline_order + 1))
        Rytheta = csr_matrix(sl * discrete_spline(ls, 2 * self.spline_order + 1))
        Rthetatheta = csr_matrix(
            -1 / sm * discrete_spline_second_derivative(ms, 2 * self.spline_order + 1)
        )

        Rxi = (
            kron(kron(RIx, RIy), RItheta)
            + kron(kron(RIIx, RIIy), RIItheta)
            + kron(kron(RIIIx, RIIIy), RIIItheta)
            + kron(kron(RIVx, RIVy), RIVtheta)
        )

        Reta = (
            kron(kron(RIIx, RIIy), RIVtheta)
            - kron(kron(RIIx, RIIy), RIItheta)
            - kron(kron(RIIIx, RIIIy), RIIItheta)
            + kron(kron(RIVx, RIVy), RItheta)
        )

        Rtheta = kron(kron(Rxtheta, Rytheta), Rthetatheta)

        R = self.Dxi * Rxi + self.Deta * Reta + self.Dtheta * Rtheta

        return R

    def _util_spline(self, theta, m1, m2):
        _, _, _, _, _, _, _, _, _, sm = self._get_dims()
        Bm1 = discrete_spline(theta / sm - m1, self.spline_order)
        Bm2 = discrete_spline(theta / sm - m2, self.spline_order)

        return np.outer(Bm1, Bm2)

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
