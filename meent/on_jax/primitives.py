import jax
import jax.numpy as jnp


def _eig(X, type_complex=jnp.complex128):
    # _eig = lambda x: jax.jit(jnp.linalg.eig, device=jax.devices('cpu')[0])(x)

    __eig = jax.jit(jnp.linalg.eig, device=jax.devices('cpu')[0])

    eigenvalues_shape = jax.ShapeDtypeStruct(X.shape[:-1], type_complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(X.shape, type_complex)

    result_shape_dtype = (eigenvalues_shape, eigenvectors_shape)

    return jax.pure_callback(__eig, result_shape_dtype, X)


@jax.custom_vjp
def eig(x):
    return _eig(x)


def eig_fwd(x):
    return eig(x), eig(x)


def eig_bwd(res, g):
    """
    Gradient of a general square (complex valued) matrix
    Reference: https://github.com/kch3782/torcwa and https://github.com/weiliangjinca/grcwa
    """
    eigval, eigvec = res

    grad_eigval, grad_eigvec = g

    grad_eigval = jnp.diag(grad_eigval)

    s = eigval.reshape((1, -1)) - eigval.reshape((-1, 1))

    # TODO: perturbation?
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
