import numpy as np
import torch


class Eig(torch.autograd.Function):
    perturbation = 1E-10

    @staticmethod
    def forward(matrix):
        res = torch.linalg.eig(matrix)
        # print(111, res[0][0], res[1][0,0])
        return res

    @staticmethod
    def setup_context(ctx, inputs, output):
        matrix, = inputs
        eigval, eigvec = output
        ctx.save_for_backward(matrix, eigval, eigvec)

    @staticmethod
    def backward(ctx, grad_eigval, grad_eigvec):
        """
        Gradient of a general square (complex valued) matrix
        Eq 2~5 in https://www.nature.com/articles/s42005-021-00568-6
        Eq 4.77 in https://arxiv.org/pdf/1701.00392.pdf
        Eq. 30~32 in https://www.sciencedirect.com/science/article/abs/pii/S0010465522002715
        https://github.com/kch3782/torcwa
        https://github.com/weiliangjinca/grcwa
        https://github.com/pytorch/pytorch/issues/41857
        https://discuss.pytorch.org/t/autograd-on-complex-numbers/144687/3
        """

        matrix, eig_val, eig_vector = ctx.saved_tensors

        grad_eigval = torch.diag(grad_eigval)
        W_H = eig_vector.T.conj()

        Fij = eig_val.reshape((1, -1)) - eig_val.reshape((-1, 1))
        Fij = Fij / (torch.abs(Fij) ** 2 + Eig.perturbation)
        diag_indices = torch.arange(len(Fij))
        Fij[diag_indices, diag_indices] = 0

        grad = torch.linalg.inv(W_H) @ (grad_eigval + Fij * (W_H @ grad_eigvec)) @ W_H
        if not torch.is_complex(matrix):
            grad = grad.real

        return grad


class RobustPinv(torch.autograd.Function):
    """
    Square-matrix pseudo-inverse with a numpy fallback for cases where
    torch's SVD (LAPACK gesdd) fails to converge on near-degenerate
    matrices (e.g. at Rayleigh/Wood's anomaly wavelengths, or at
    intermediate designs during topology optimization). numpy's pinv uses
    a different LAPACK driver (gelsd) that is far more robust on these
    ill-conditioned inputs.

    Every call site in this codebase applies meeinv to a square matrix
    (mode matrix W/V, or A/B/Q), where pinv is used purely as a
    numerically-safe stand-in for inv, not as a true rectangular
    least-squares pseudo-inverse. So the backward pass uses the standard
    square-matrix inverse VJP, d(A^-1) = -A^-1 dA A^-1, which is exact
    for the (near-)invertible case regardless of which algorithm computed
    the forward result. This keeps gradients flowing even through
    wavelengths/designs where the numpy fallback is triggered, which
    matters for gradient-based inverse design.
    """

    @staticmethod
    def forward(x):
        try:
            res = torch.linalg.pinv(x)
        except torch.linalg.LinAlgError:
            x_np = x.detach().cpu().numpy()
            res_np = np.linalg.pinv(x_np)
            res = torch.as_tensor(res_np, dtype=x.dtype, device=x.device)
        return res

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        res, = ctx.saved_tensors
        res_h = res.conj().transpose(-2, -1)
        grad_input = -res_h @ grad_output @ res_h
        return grad_input


def meeinv(x, use_pinv=False):
    if use_pinv:
        res = RobustPinv.apply(x)
    else:
        try:
            res = torch.linalg.inv(x)
        except torch.linalg.LinAlgError:
            # Singular matrix under plain inv (e.g. a Rayleigh/Wood's anomaly
            # wavelength). Fall back to the robust pinv automatically instead
            # of crashing, so callers don't need to pass use_pinv=True
            # themselves or know in advance which inputs will be singular.
            res = RobustPinv.apply(x)

    return res
