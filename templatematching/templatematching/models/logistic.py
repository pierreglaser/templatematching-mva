import numpy as np
from scipy.special import expit
from templatematching.models.linear import R2Ridge


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
        max_iter=10,
        tol=1e-8,
        random_state=None,
        eye="left",
        verbose=0,
    ):
        super().__init__(
            template_shape,
            splines_per_axis,
            spline_order,
            mu,
            verbose,
            random_state=random_state,
            eye=eye
        )
        self.model_name = "Logistic Ridge"
        self.max_iter = max_iter
        self.tol = tol
        self.rs = np.random.RandomState(random_state)

    def _fit_patches(self, X, y):
        self._check_params(X)
        y = y.reshape(-1, 1)

        Nk, Nl = self.splines_per_axis
        S = self._create_s_matrix(X)

        # Randomly intialize c
        c = self.rs.rand(Nk * Nl, 1)

        # Normalize c
        c /= np.linalg.norm(c)

        loss = 1

        # Optimization step
        for iter_no in range(self.max_iter):

            if loss < self.tol:
                break

            c_old = c.copy()
            p = expit(S @ c)
            w = p * (1 - p)

            W = np.diag(w.flatten())

            # Close form hessian
            H = -(S.T @ W @ S + self.mu * np.eye(S.shape[1]))
            H_inv = np.linalg.inv(H + self.tol * np.eye(H.shape[0]))

            # Close form gradient
            grad = S.T @ (y - p) - self.mu * np.eye(S.shape[1]) @ c
            c -= H_inv @ grad
            c /= np.linalg.norm(c)

            loss = np.linalg.norm(c - c_old)

            if self.verbose:
                print(f"iteration no: {iter_no}, change in coefs: {loss}")
        self._S, self._spline_coef = S, c
