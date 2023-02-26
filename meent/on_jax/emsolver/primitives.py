import jax
import jax.numpy as jnp

from functools import partial


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def eig(x, type_complex=jnp.complex128, perturbation=1E-10, device='cpu'):

    _eig = jax.jit(jnp.linalg.eig, device=jax.devices('cpu')[0])

    eigenvalues_shape = jax.ShapeDtypeStruct(x.shape[:-1], type_complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(x.shape, type_complex)

    result_shape_dtype = (eigenvalues_shape, eigenvectors_shape)
    if device == 'cpu':
        res = _eig(x)
    else:
        res = jax.pure_callback(_eig, result_shape_dtype, x)

    return res


def eig_fwd(x, type_complex, perturbation, device):
    return eig(x, type_complex, perturbation), eig(x, type_complex, perturbation)


def eig_bwd(type_complex, perturbation, device, res, g):
    """
    Gradient of a general square (complex valued) matrix
    Reference: https://github.com/kch3782/torcwa and https://github.com/weiliangjinca/grcwa
    """
    eigval, eigvec = res

    grad_eigval, grad_eigvec = g

    grad_eigval = jnp.diag(grad_eigval)

    s = eigval.reshape((1, -1)) - eigval.reshape((-1, 1))

    F = s / (jnp.abs(s) ** 2 + perturbation)
    F = F.at[jnp.diag_indices_from(s)].set(0)

    XH = eigvec.T
    tmp = F * (XH @ grad_eigvec)

    XH_i = jnp.linalg.inv(XH)

    grad = (XH_i @ (grad_eigval + tmp)) @ XH

    if not jnp.iscomplexobj(eigval):
        grad = grad.real

    return grad,


eig.defvjp(eig_fwd, eig_bwd)
