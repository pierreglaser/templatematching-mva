import numpy as np
from scipy.special import expit
from templatematching.models.linear import R2Ridge, SE2Ridge


# XXX: R2LogReg inheriting from R2Ridge is a hack: the commom
# logic between the two classes should be factored out into the
# Spline Base class or a Spline Mixin Class
class R2LogReg(R2Ridge):
    def __init__(
        self,
        template_shape,
        splines_per_axis,
        spline_order=2,
        mu=0,
        lbd=0,
        max_iter=10,
        batch_size=50,
        tol=1e-8,
        random_state=None,
        early_stopping=1,
        eye="left",
        verbose=0,
    ):
        super().__init__(
            template_shape,
            splines_per_axis,
            spline_order,
            batch_size,
            mu,
            lbd,
            verbose,
            random_state=random_state,
            eye=eye,
        )
        self.model_name = "Logistic Ridge"
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.rs = np.random.RandomState(random_state)

    def _fit_patches(self, X, y):
        self._check_params(X)
        num_samples, _, _, Nk, Nl, _, _ = self._get_dims()
        y = y.reshape(-1, 1)

        S = self._create_s_matrix(X)
        R = self._create_r_matrix()

        # Randomly intialize c
        c = self.rs.rand(Nk * Nl, 1)

        # Normalize c
        c /= np.linalg.norm(c)

        loss = 1e7
        count_loss_evol = 0
        loss_prev = loss
        # Optimization step
        for iter_no in range(self.max_iter):

            if loss < self.tol:
                break

            c_old = c.copy()
            p = expit(S @ c)
            w = p * (1 - p)

            W = np.diag(w.flatten())

            # Close form hessian
            H = -(
                S.T @ W @ S
                + num_samples * (self.lbd * R + self.mu * np.eye(S.shape[1]))
            )
            H_inv = np.linalg.inv(H + self.tol * np.eye(H.shape[0]))

            # Close form gradient
            grad = (
                S.T @ (y - p)
                - num_samples * self.lbd * R @ c
                - num_samples * self.mu * np.eye(S.shape[1]) @ c
            )
            c -= H_inv @ grad
            c /= np.linalg.norm(c)

            loss = np.linalg.norm(c - c_old)

            if self.verbose:
                print(f"iteration no: {iter_no}, change in coefs: {loss}")

            if loss_prev - loss < 0:
                count_loss_evol += 1

            else:
                count_loss_evol = 0
                loss_prev = loss
                best_c = c.copy()

            if count_loss_evol > self.early_stopping:
                if self.verbose:
                    print(f"Loss did not improve in {self.early_stopping} iteration(s)")
                break

        self._S, self._spline_coef = S, best_c


class SE2LogReg(SE2Ridge):
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
        max_iter=10,
        tol=1e-8,
        early_stopping=10,
        random_state=None,
        eye="left",
        verbose=0,
    ):
        super().__init__(
            template_shape,
            splines_per_axis,
            wavelet_dim,
            num_orientation_slices,
            spline_order,
            batch_size,
            mu,
            lbd,
            Dxi,
            Deta,
            Dtheta,
            verbose,
            random_state=random_state,
            eye=eye,
        )

        self.model_name = "SE2 Logistic Ridge"
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.rs = np.random.RandomState(random_state)

    def _fit_patches(self, X, y):
        X = abs(self._ost.fit_transform(X))  # can also take the imag
        self._check_params(X)
        y = y.reshape(-1, 1)

        num_samples, _, _, _, Nk, Nl, Nm, _, _, _ = self._get_dims()
        S = self._create_s_matrix(X)
        R = self._create_r_matrix()

        # Randomly intialize c
        c = self.rs.rand(Nk * Nl * Nm, 1)

        # Normalize c
        c /= np.linalg.norm(c)

        loss = 1e7
        count_loss_evol = 0
        loss_prev = loss
        # Optimization step
        for iter_no in range(self.max_iter):

            if loss < self.tol:
                break

            c_old = c.copy()
            p = expit(S @ c)
            w = p * (1 - p)

            W = np.diag(w.flatten())

            # Close form hessian
            H = -(
                S.T @ W @ S
                + num_samples * (self.lbd * R + self.mu * np.eye(S.shape[1]))
            )
            H_inv = np.linalg.inv(H + self.tol * np.eye(H.shape[0]))

            # Close form gradient
            grad = (
                S.T @ (y - p)
                - num_samples * self.lbd * R @ c
                - num_samples * self.mu * np.eye(S.shape[1]) @ c
            )
            c -= H_inv @ grad
            c /= np.linalg.norm(c)

            loss = np.linalg.norm(c - c_old)

            if self.verbose:
                print(f"iteration no: {iter_no}, change in coefs: {loss}")

            if loss_prev - loss < 0:
                count_loss_evol += 1

            else:
                count_loss_evol = 0
                loss_prev = loss
                best_c = c.copy()

            if count_loss_evol > self.early_stopping:
                if self.verbose:
                    print(f"Loss did not improve in {self.early_stopping} iteration(s)")
                break

        self._S, self._spline_coef = S, best_c
