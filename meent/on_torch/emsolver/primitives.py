import numpy as np
import torch


class Eig(torch.autograd.Function):
    """torch.linalg.eig with a backward pass that survives degenerate eigenvalues.

    The analytic gradient of an eigendecomposition carries a 1 / (lambda_i - lambda_j) factor
    off the diagonal, which blows up wherever two eigenvalues coincide. RCWA hits that
    routinely - a homogeneous layer has every eigenvalue repeated - so torch's own backward
    returns nan there. `perturbation` softens the pole instead of dividing by it.
    """

    perturbation = 1E-10

    @staticmethod
    def forward(matrix):
        res = torch.linalg.eig(matrix)
        return res

    @staticmethod
    def setup_context(ctx, inputs, output):
        matrix, = inputs
        eigval, eigvec = output
        ctx.save_for_backward(matrix, eigval, eigvec)

    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvec):
        matrix, eig_val, eig_vector = ctx.saved_tensors

        grad_eigval = torch.diag(grad_eigval)
        W_H = eig_vector.T.conj()

        # Fij is 1 / (lambda_j - lambda_i) written so the pole is bounded: multiplying the
        # difference by its own conjugate turns 1/d into conj(d)/(|d|^2 + eps). At eps = 0
        # this is exactly 1/d; near a degeneracy it saturates instead of diverging, which
        # costs accuracy in the gradient there but keeps it finite.
        Fij = eig_val.reshape((1, -1)) - eig_val.reshape((-1, 1))
        Fij = Fij / (torch.abs(Fij) ** 2 + Eig.perturbation)
        # The diagonal is the eigenvalue term, already carried by grad_eigval; zero it so it
        # is not counted twice. (d = 0 there, so the entry would be 0 anyway once eps > 0 -
        # this is explicit rather than relying on that.)
        diag_indices = torch.arange(len(Fij), device=Fij.device)
        Fij[diag_indices, diag_indices] = 0

        grad = torch.linalg.inv(W_H) @ (grad_eigval + Fij * (W_H @ grad_eigvec)) @ W_H
        # A real input must get a real gradient back; the decomposition itself is complex even
        # when the matrix is not, so the imaginary part here is an artefact of the route taken.
        if not torch.is_complex(matrix):
            grad = grad.real

        return grad


class RobustPinv(torch.autograd.Function):
    """Pseudo-inverse that falls back to numpy, with the square-inverse gradient.

    Both libraries reach the pseudo-inverse through an SVD, and the SVD is iterative: it can
    fail to converge on one implementation and succeed on the other for the same matrix. The
    fallback is there for that case only, not for a difference in the answer.
    """

    @staticmethod
    def forward(x):
        try:
            res = torch.linalg.pinv(x)
        except torch.linalg.LinAlgError:
            # Leaving the device for the fallback is deliberate; a GPU SVD that has already
            # failed is not worth retrying on the GPU.
            x_np = x.detach().cpu().numpy()
            res_np = np.linalg.pinv(x_np)
            res = torch.as_tensor(res_np, dtype=x.dtype, device=x.device)
        return res

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        # d(A^-1) = -A^-1 dA A^-1, which is the gradient of a true inverse. The pseudo-inverse
        # of a rank-deficient matrix has two further terms, living in the null spaces that a
        # true inverse does not have. They are dropped here: the matrices this is called on
        # are meant to be invertible, and pinv is the safety net rather than the intent. On a
        # genuinely singular matrix the forward value is still right and the gradient is not.
        res, = ctx.saved_tensors
        res_h = res.conj().transpose(-2, -1)
        grad_input = -res_h @ grad_output @ res_h
        return grad_input


def meeinv(x, use_pinv=False):
    """Invert x, keeping the graph. `use_pinv` asks for the pseudo-inverse from the start.

    Without it the plain inverse is tried first and pinv is only the fallback, so a matrix
    that is merely ill-conditioned - not singular - still goes the cheaper, exact route.
    """
    if use_pinv:
        res = RobustPinv.apply(x)
    else:
        try:
            res = torch.linalg.inv(x)
        except torch.linalg.LinAlgError:
            res = RobustPinv.apply(x)

    return res
