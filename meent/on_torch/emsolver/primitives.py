import numpy as np
import torch


class Eig(torch.autograd.Function):
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

        Fij = eig_val.reshape((1, -1)) - eig_val.reshape((-1, 1))
        Fij = Fij / (torch.abs(Fij) ** 2 + Eig.perturbation)
        diag_indices = torch.arange(len(Fij), device=Fij.device)
        Fij[diag_indices, diag_indices] = 0

        grad = torch.linalg.inv(W_H) @ (grad_eigval + Fij * (W_H @ grad_eigvec)) @ W_H
        if not torch.is_complex(matrix):
            grad = grad.real

        return grad


class RobustPinv(torch.autograd.Function):
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
            res = RobustPinv.apply(x)

    return res
