import numpy as np

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
        self.batch_size = batch_size
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

        for i in range(int(num_samples / self.batch_size)):
            print(i)
            X_batch = X[i * self.batch_size:(i+1) * self.batch_size, :, :]
            convolved_X = fftconvolve(X_batch, B, mode="same")
            S[i * self.batch_size:(i+1) * self.batch_size, :] = convolved_X[:, ::sk, ::sl].reshape(self.batch_size, Nk * Nl)
       
        S /= np.linalg.norm(S, axis=0, keepdims=True)

        return S

    def _create_r_matrix(self):
        _, _, _, Nk, Nl, sk, sl = self._get_dims()
        k = np.linspace(-int(Nk / 2), int(Nk / 2), Nk)
        l = np.linspace(-int(Nl / 2), int(Nl / 2), Nl)

        ks = np.array([[ki - kk for ki in k] for kk in k])
        ls = np.array([[li - lk for li in l] for lk in l])

        Bxk = -1 / sk * discrete_spline_second_derivative(ks, 2 * self.spline_order + 1)
        Bxl = sl * discrete_spline(ls, 2 * self.spline_order + 1)
        Byk = sk * discrete_spline(ks, 2 * self.spline_order + 1)
        Byl = -1 / sl * discrete_spline_second_derivative(ls, 2 * self.spline_order + 1)

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
        batch_size=50,
        mu=0,
        lbd=0,
        Dxi=0,
        Deta=0,
        Dtheta=0,
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
            self, patch_shape=template_shape, eye=eye, random_state=random_state
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
        num_samples, _, _, _, _, _, _, _, _, _ = self._get_dims()
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
                B_inv = np.linalg.inv(B)
                c = (
                    B_inv
                    @ S.T
                    @ np.linalg.inv(S @ B_inv @ S.T + np.eye(S.shape[0]))
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

        for i in range(int(num_samples / self.batch_size)):
            print(i)
            X_batch = X[i * self.batch_size:(i+1) * self.batch_size, :, :]
            print(X_batch.shape, B.shape)
            convolved_X = fftconvolve(X_batch, B, mode="same")
            S[i * self.batch_size:(i+1) * self.batch_size, :] = convolved_X[:, ::sk, ::sl, ::sm].reshape(self.batch_size, Nk * Nl)

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

        RIx = -1 / sk * discrete_spline_second_derivative(ks, 2 * self.spline_order + 1)
        RIy = sl * discrete_spline(ls, 2 * self.spline_order + 1)
        cos2_spline = lambda theta, m1, m2: np.cos(theta) ** 2 * self._util_spline(
            theta, m1, m2
        )
        RItheta = np.array(
            [[quad(cos2_spline, 0, np.pi, args=(m1, m2))[0] for m1 in m] for m2 in m]
        )

        RIIx = discrete_spline_second_derivative(ks, 2 * self.spline_order + 1)
        RIIy = -discrete_spline_second_derivative(ls, 2 * self.spline_order + 1)
        sincos_spline = (
            lambda theta, m1, m2: np.cos(theta)
            * np.sin(theta)
            * self._util_spline(theta, m1, m2)
        )
        RIItheta = np.array(
            [[quad(sincos_spline, 0, np.pi, args=(m1, m2))[0] for m1 in m] for m2 in m]
        )

        RIIIx = -RIIx.copy()
        RIIIy = -RIIy.copy()
        RIIItheta = RIItheta.copy()

        RIVx = sk * discrete_spline(ks, 2 * self.spline_order + 1)
        RIVy = (
            -1 / sl * discrete_spline_second_derivative(ls, 2 * self.spline_order + 1)
        )
        sin2_spline = lambda theta, m1, m2: np.sin(theta) ** 2 * self._util_spline(
            theta, m1, m2
        )
        RIVtheta = np.array(
            [[quad(sin2_spline, 0, np.pi, args=(m1, m2))[0] for m1 in m] for m2 in m]
        )

        Rxtheta = sk * discrete_spline(ks, 2 * self.spline_order + 1)
        Rytheta = sl * discrete_spline(ls, 2 * self.spline_order + 1)
        Rthetatheta = (
            -1 / sm * discrete_spline_second_derivative(ms, 2 * self.spline_order + 1)
        )

        Rxi = (
            np.kron(np.kron(RIx, RIy), RItheta)
            + np.kron(np.kron(RIIx, RIIy), RIItheta)
            + np.kron(np.kron(RIIIx, RIIIy), RIIItheta)
            + np.kron(np.kron(RIVx, RIVy), RIVtheta)
        )

        Reta = (
            np.kron(np.kron(RIIx, RIIy), RIVtheta)
            - np.kron(np.kron(RIIx, RIIy), RIItheta)
            - np.kron(np.kron(RIIIx, RIIIy), RIIItheta)
            + np.kron(np.kron(RIVx, RIVy), RItheta)
        )

        Rtheta = np.kron(np.kron(Rxtheta, Rytheta), Rthetatheta)

        return self.Dxi * Rxi + self.Deta * Reta + self.Dtheta * Rtheta

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
