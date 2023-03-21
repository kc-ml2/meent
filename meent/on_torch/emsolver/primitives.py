"""
Gradient of a general square (complex valued) matrix
Eq. 30~32 in https://www.sciencedirect.com/science/article/abs/pii/S0010465522002715
Eq 4.77 in https://arxiv.org/pdf/1701.00392.pdf
https://github.com/kch3782/torcwa
https://github.com/weiliangjinca/grcwa
"""

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

        grad_eigval = grad_eigval.conj()
        grad_eigvec = grad_eigvec.conj()

        matrix, eig_val, eig_vector = ctx.saved_tensors

        grad_eigval = torch.diag(grad_eigval)
        X_H = eig_vector.T.conj()

        Fij = eig_val.conj().reshape((1, -1)) - eig_val.conj().reshape((-1, 1))
        Fij = Fij / (torch.abs(Fij) ** 2 + Eig.perturbation)
        diag_indices = torch.arange(len(Fij))
        Fij[diag_indices, diag_indices] = 0

        grad = torch.linalg.inv(X_H) @ (grad_eigval.conj() + Fij.conj() * (X_H @ grad_eigvec.conj())) @ X_H
        if not torch.is_complex(matrix):
            grad = grad.real

        return grad
