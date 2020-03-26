import numpy as np
from scipy.special import expit
from templatematching.models.linear import R2Ridge


class R2LogReg(R2Ridge):
    def __init__(self, template_shape, spline_order=2, mu=0, optimizer_steps=10, random_state=None, verbose=0):
        super().__init__(template_shape, spline_order, mu, verbose)
        self.optimizer_steps = optimizer_steps
        self.rs = np.random.RandomState(random_state)

    def fit(self, X, y):

        TOL = 1E-4
        Nk, Nl = self.template_shape
        S = self._make_s_matrix(X)

        # Randomly intialize c
        c = self.rs.rand(Nk * Nl, 1)

        # Normalize c
        c /= np.linalg.norm(c)

        loss = 1

        # Optimization step
        for _ in range(self.optimizer_steps):

            if loss < TOL:
                break

            c_old = c.copy()
            p = expit(S @ c)
            w = p * (1 - p)

            W = np.diag(w.flatten())

            # Close form hessian
            H = - (S.T @ W @ S + self.mu * np.eye(S.shape[1]))
            H_inv = np.linalg.inv(H +  TOL * np.eye(H.shape[0]))

            # Close form gradient
            grad  = S.T @ (y - p) - self.mu * np.eye(S.shape[1]) @ c
            #grad = S.T @ W @ (S @ c + W_inv @ (y - p))
            c -= H_inv @ grad
            c /= np.linalg.norm(c)

            loss = np.linalg.norm(c - c_old)

            if self.verbose:
                print(f"Loss: {loss}")
        self._S, self._Nx, self._Ny = S, X.shape[1], X.shape[2]
        self.spline_coef = c
        self._template = self.reconstruct_template()
