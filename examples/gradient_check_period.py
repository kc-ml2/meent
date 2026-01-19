import torch
import jax
import jax.numpy as jnp
import numpy as np
import meent

def get_ucell(mode, backend):
    # simple pattern
    # 1D: (1, 1, 10)
    # 2D: (1, 10, 10)
    if mode == '1D':
        # 1 layer, y=1, x=10
        ucell = np.array([[[1.0]*5 + [1.5]*5]]) 
    else: # 2D
        # 1 layer, y=10, x=10
        x = np.array([[1.0]*5 + [1.5]*5])
        ucell = np.repeat(x, 10, axis=0).reshape(1, 10, 10)
        
    if backend == 1: # JAX
        return jnp.array(ucell)
    else: # Torch
        return torch.tensor(ucell, dtype=torch.float64)

def run_jax(mode, shape_type):
    print(f"  Testing JAX, mode={mode}, shape={shape_type}")
    
    ucell = get_ucell(mode, 1)
    
    def loss_fn(p):
        # p is whatever structure jax.grad passes down (list of tracers, array tracer, etc.)
        fto = [5, 0] if mode == '1D' else [5, 5]
        mee = meent.call_mee(backend=1, period=p, ucell=ucell,
                             wavelength=900., thickness=[500.], fto=fto,
                             n_top=1., n_bot=1.5)
        res = mee.conv_solve()
        return jnp.sum(res.res.de_ri)

    val = 1000.
    
    if shape_type == 'scalar':
        p_init = val
    elif shape_type == 'vector_1':
        p_init = jnp.array([val])
    elif shape_type == 'vector_2':
        p_init = jnp.array([val, val])
    elif shape_type == 'list_1':
        p_init = [val]
    elif shape_type == 'list_2':
        p_init = [val, val]
        
    # JAX Grad
    # For list input, we need to ensure jax treats it correctly. 
    # jax.grad differentiates w.r.t first argument.
    grad_fn = jax.grad(loss_fn)
    
    try:
        ad_grad = grad_fn(p_init)
    except Exception as e:
        print(f"    FAILED AD: {e}")
        # import traceback
        # traceback.print_exc()
        return

    # Finite Difference
    eps = 1e-4
    
    # Helper to compute FD
    def compute_loss_val(p_in):
        # We need to wrap scalar in array if needed for consistent loss_fn calls?
        # No, loss_fn handles what we pass.
        return loss_fn(p_in)

    fd_grad = None

    if shape_type == 'scalar':
         l_plus = compute_loss_val(p_init + eps)
         l_minus = compute_loss_val(p_init - eps)
         fd_grad = (l_plus - l_minus) / (2 * eps)
         
    elif shape_type.startswith('vector'):
         p_val = np.array(p_init)
         grad_flat = []
         for i in range(p_val.size):
             p_plus = p_val.flatten()
             p_plus[i] += eps
             p_plus = p_plus.reshape(p_val.shape)
             
             p_minus = p_val.flatten()
             p_minus[i] -= eps
             p_minus = p_minus.reshape(p_val.shape)

             l_p = compute_loss_val(jnp.array(p_plus))
             l_m = compute_loss_val(jnp.array(p_minus))
             grad_flat.append((l_p - l_m) / (2*eps))
         fd_grad = np.array(grad_flat).reshape(p_val.shape)
         
    elif shape_type.startswith('list'):
         grad_list = []
         for i in range(len(p_init)):
             p_plus = list(p_init)
             p_plus[i] += eps
             p_minus = list(p_init)
             p_minus[i] -= eps
             l_p = compute_loss_val(p_plus)
             l_m = compute_loss_val(p_minus)
             grad_list.append((l_p - l_m)/(2*eps))
         fd_grad = grad_list

    # Compare
    if isinstance(ad_grad, list):
        ad_arr = np.array(ad_grad)
        fd_arr = np.array(fd_grad)
    else:
        ad_arr = np.array(ad_grad)
        fd_arr = np.array(fd_grad)

    diff = np.abs(ad_arr - fd_arr).max()
    if diff < 1e-4:
        print(f"    PASSED (Diff: {diff:.2e})")
        print(f"    AD: {ad_grad}")
        print(f"    FD: {fd_grad}")
    else:
        print(f"    FAILED (Diff: {diff:.2e})")
        print(f"    AD: {ad_grad}")
        print(f"    FD: {fd_grad}")

def run_torch(mode, shape_type):
    print(f"  Testing Torch, mode={mode}, shape={shape_type}")
    
    ucell = get_ucell(mode, 2)
    
    val = 1000.
    
    if shape_type == 'scalar':
        p = torch.tensor(val, dtype=torch.float64, requires_grad=True)
    elif shape_type == 'vector_1':
        p = torch.tensor([val], dtype=torch.float64, requires_grad=True)
    elif shape_type == 'vector_2':
        p = torch.tensor([val, val], dtype=torch.float64, requires_grad=True)
    elif shape_type == 'list_1':
        p = [torch.tensor(val, dtype=torch.float64, requires_grad=True)]
    elif shape_type == 'list_2':
        p = [torch.tensor(val, dtype=torch.float64, requires_grad=True) for _ in range(2)]

    def loss_fn(p_in):
        fto = [5, 0] if mode == '1D' else [5, 5]
        mee = meent.call_mee(backend=2, period=p_in, ucell=ucell,
                             wavelength=900., thickness=[500.], fto=fto,
                             n_top=1., n_bot=1.5)
        res = mee.conv_solve()
        return res.res.de_ri.sum()

    # AD
    try:
        loss = loss_fn(p)
        loss.backward()
    except Exception as e:
        print(f"    FAILED AD: {e}")
        return
    
    if isinstance(p, list):
        ad_grad = [pi.grad.item() for pi in p]
    else:
        ad_grad = p.grad.detach().numpy()
        
    # FD
    eps = 1e-4
    
    def compute_loss_val_torch(p_vals):
        # p_vals is list of floats or scalar float or array
        # we need to reconstruct input
        if shape_type == 'scalar':
             return loss_fn(torch.tensor(p_vals, dtype=torch.float64))
        elif shape_type.startswith('vector'):
             return loss_fn(torch.tensor(p_vals, dtype=torch.float64))
        elif shape_type.startswith('list'):
             # pass list of tensors (constants)
             return loss_fn([torch.tensor(v, dtype=torch.float64) for v in p_vals])
    
    fd_grad = None
    
    with torch.no_grad():
        if shape_type == 'scalar':
             p_val = p.item()
             l_p = compute_loss_val_torch(p_val + eps)
             l_m = compute_loss_val_torch(p_val - eps)
             fd_grad = (l_p - l_m).item()/(2*eps)
             
        elif shape_type.startswith('vector'):
             p_val = p.detach().numpy()
             grad_flat = []
             for i in range(p_val.size):
                 p_plus = p_val.flatten()
                 p_plus[i] += eps
                 p_plus = p_plus.reshape(p_val.shape)
                 
                 p_minus = p_val.flatten()
                 p_minus[i] -= eps
                 p_minus = p_minus.reshape(p_val.shape)
                 
                 l_p = compute_loss_val_torch(p_plus)
                 l_m = compute_loss_val_torch(p_minus)
                 grad_flat.append((l_p - l_m).item()/(2*eps))
             fd_grad = np.array(grad_flat).reshape(p_val.shape)
             
        elif shape_type.startswith('list'):
             grad_list = []
             p_vals = [pi.item() for pi in p]
             for i in range(len(p_vals)):
                 p_plus = list(p_vals)
                 p_plus[i] += eps
                 
                 p_minus = list(p_vals)
                 p_minus[i] -= eps
                 
                 l_p = compute_loss_val_torch(p_plus)
                 l_m = compute_loss_val_torch(p_minus)
                 grad_list.append((l_p - l_m).item()/(2*eps))
             fd_grad = grad_list

    # Compare
    ad_arr = np.array(ad_grad)
    fd_arr = np.array(fd_grad)
    diff = np.abs(ad_arr - fd_arr).max()
    
    if diff < 1e-4:
        print(f"    PASSED (Diff: {diff:.2e})")
        print(f"    AD: {ad_grad}")
        print(f"    FD: {fd_grad}")
    else:
        print(f"    FAILED (Diff: {diff:.2e})")
        print(f"    AD: {ad_grad}")
        print(f"    FD: {fd_grad}")

if __name__ == '__main__':
    print("========================================")
    print("Gradient Check for Period Parameter")
    print("========================================")
    
    shapes = ['scalar', 'vector_1', 'vector_2', 'list_1', 'list_2']
    modes = ['1D', '2D']
    
    print("\n--- JAX Backend ---")
    for mode in modes:
        for shape in shapes:
            run_jax(mode, shape)
            
    print("\n--- PyTorch Backend ---")
    for mode in modes:
        for shape in shapes:
            run_torch(mode, shape)
