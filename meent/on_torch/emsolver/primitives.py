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
