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
    return eig(x, type_complex, perturbation), (eig(x, type_complex, perturbation), x)


def eig_bwd(type_complex, perturbation, device, res, g):
    """
    Gradient of a general square (complex valued) matrix
    Eq. 30~32 in https://www.sciencedirect.com/science/article/abs/pii/S0010465522002715
    Eq 4.77 in https://arxiv.org/pdf/1701.00392.pdf
    https://github.com/kch3782/torcwa
    https://github.com/weiliangjinca/grcwa
    """

    (eig_val, eig_vector), x = res
    grad_eigval, grad_eigvec = g
    grad_eigval = jnp.diag(grad_eigval)

    X_H = eig_vector.T.conj()

    Fij = eig_val.conj().reshape((1, -1)) - eig_val.conj().reshape((-1, 1))
    Fij = Fij / (jnp.abs(Fij) ** 2 + perturbation)
    Fij = Fij.at[jnp.diag_indices_from(Fij)].set(0)
    grad = jnp.linalg.inv(X_H) @ (grad_eigval.conj() + Fij.conj() * (X_H @ grad_eigvec.conj())) @ X_H
    grad = grad.conj()
    if not jnp.iscomplexobj(x):
        grad = grad.real
    return grad,


eig.defvjp(eig_fwd, eig_bwd)
