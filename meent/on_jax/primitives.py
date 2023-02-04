import jax
import jax.numpy as jnp


def _eig(X, type_complex=jnp.complex128):
    _eig = lambda x: jax.jit(jnp.linalg.eig, device=jax.devices('cpu')[0])(x)

    eigenvalues_shape = jax.ShapeDtypeStruct(X.shape[:-1], type_complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(X.shape, type_complex)

    result_shape_dtype = (eigenvalues_shape, eigenvectors_shape)

    return jax.pure_callback(_eig, result_shape_dtype, X)


@jax.custom_vjp
def eig(x):
    # return jnp.linalg.eig(x)
    return _eig(x)


def eig_fwd(x):
    return eig(x), eig(x)


def eig_bwd(res, g):
    """
    Gradient of a general square (complex valued) matrix
    Reference: https://github.com/kch3782/torcwa and https://github.com/weiliangjinca/grcwa
    """
    eigval, eigvec = res  # eigenvalues as 1d array, eigenvectors in columns

    grad_eigval, grad_eigvec = g

    grad_eigval = jnp.diag(grad_eigval)

    s = eigval.reshape((1, -1)) - eigval.reshape((-1, 1))

    F = jnp.conj(s) / (jnp.abs(s) ** 2 + 1E-10)
    F = F.at[jnp.diag_indices_from(s)].set(0)

    XH = jnp.conj(eigvec).T
    tmp = jnp.conj(F) * (XH @ grad_eigvec)

    XH_i = jnp.linalg.inv(XH)

    grad = (XH_i @ (grad_eigval + tmp)) @ XH

    if not jnp.iscomplexobj(eigval):
        grad = grad.real

    return grad,


eig.defvjp(eig_fwd, eig_bwd)

# def grad_inv(ans, x):
#     return lambda g: -_dot(_dot(T(ans), g), T(ans))


# @staticmethod
# def backward(ctx, grad_eigval, grad_eigvec):
#
#     import torch
#
#     eigval = ctx.eigval.to(grad_eigval)
#     eigvec = ctx.eigvec.to(grad_eigvec)
#
#     grad_eigval = torch.diag(grad_eigval)
#     s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)
#
#     # Lorentzian broadening: get small error but stabilizing the gradient calculation
#     if Eig.broadening_parameter is not None:
#         F = torch.conj(s) / (torch.abs(s) ** 2 + Eig.broadening_parameter)
#     elif s.dtype == torch.complex64:
#         F = torch.conj(s) / (torch.abs(s) ** 2 + 1.4e-45)
#     elif s.dtype == torch.complex128:
#         F = torch.conj(s) / (torch.abs(s) ** 2 + 4.9e-324)
#
#     diag_indices = torch.linspace(0, F.shape[-1] - 1, F.shape[-1], dtype=torch.int64)
#     F[diag_indices, diag_indices] = 0.
#     XH = torch.transpose(torch.conj(eigvec), -2, -1)
#     tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)
#
#     grad = torch.matmul(torch.matmul(torch.inverse(XH), grad_eigval + tmp), XH)
#     if not torch.is_complex(ctx.input):
#         grad = torch.real(grad)
#
#     return grad
#
# def grad_eig_old(res, g):
#     """Gradient of a general square (complex valued) matrix"""
#     eigval, eigvec = res  # eigenvalues as 1d array, eigenvectors in columns
#     n = eigval.shape[-1]
#
#     grad_eigval, grad_eigvec = g
#
#     reps = jnp.array(grad_eigval.shape)
#     reps = reps.at[:-1].set(1)
#
#     reps = reps.at[-1].set(grad_eigval.shape[-1])
#
#     newshape = list(grad_eigval.shape) + [grad_eigval.shape[-1]]
#
#     _diag = lambda a: jnp.eye(a.shape[-1]) * a
#
#     ge_new = _diag(jnp.tile(grad_eigval, reps).reshape(newshape))
#     grad_eigval = ge_new
#     ############
#
#     f = 1 / (eigval[..., jnp.newaxis, :] - eigval[..., :, jnp.newaxis] + 1.e-20)
#     f -= _diag(f)
#     ut = jnp.swapaxes(eigvec, -1, -2)
#     r1 = f * _dot(ut, grad_eigvec)
#     r2 = -f * (_dot(_dot(ut, jnp.conj(eigvec)), jnp.real(_dot(ut, grad_eigvec)) * jnp.eye(n)))
#     r = _dot(_dot(inv(ut), grad_eigval + r1 + r2), ut)
#
#     # if not jnp.iscomplexobj(x):
#     #     r = jnp.real(r)
#     # the derivative is still complex for real input (imaginary delta is allowed), real output
#     # but the derivative should be real in real input case when imaginary delta is forbidden
#     return r,
#
