import torch
import jax
import jax.numpy as jnp
import numpy as np
import meent

def get_ucell(mode, backend):
    """
    Generates the unit cell (refractive index distribution) for the simulation.
    
    Args:
        mode (str): '1D' or '2D'.
        backend (int): 1 for JAX, 2 for PyTorch.
        
    Returns:
        Tensor/Array: The unit cell structure.
    """
    # Simple pattern generation
    # 1D: (1, 1, 10) - extruded along y, varying along x
    # 2D: (1, 10, 10) - varying along both x and y
    if mode == '1D':
        # 1 layer, y=1, x=10. Simple binary grating.
        ucell = np.array([[[1.0]*5 + [1.5]*5]]) 
    else: # 2D
        # 1 layer, y=10, x=10. 
        # Checkerboard-like pattern (square in middle) to ensure sensitivity to both X and Y periods.
        # If the pattern were invariant in Y (like a 1D grating in 2D sim), dL/dTy would be 0.
        ucell = np.ones((1, 10, 10)) * 1.0
        ucell[0, 2:8, 2:8] = 1.5
        
    if backend == 1: # JAX
        return jnp.array(ucell)
    else: # Torch
        return torch.tensor(ucell, dtype=torch.float64)

def run_jax(mode, shape_type):
    """
    Runs gradient check for JAX backend.
    """
    print(f"  Testing JAX, mode={mode}, shape={shape_type}")
    
    ucell = get_ucell(mode, 1)
    
    def loss_fn(p):
        """
        Calculates total reflection efficiency.
        Jax will differentiate this function w.r.t 'p'.
        """
        # Set Fourier orders (fto) based on mode.
        # 1D: [5, 0] means 11 orders in X, 0 in Y.
        # 2D: [5, 5] means 11 orders in X, 11 in Y.
        fto = [5, 0] if mode == '1D' else [5, 5]
        
        # Call meent. 'period' is passed as 'p'.
        mee = meent.call_mee(backend=1, period=p, ucell=ucell,
                             wavelength=900., thickness=[500.], fto=fto,
                             n_top=1., n_bot=1.5)
        res = mee.conv_solve()
        
        # Loss: Sum of reflection efficiencies (Diffraction Efficiency - Reflection - Intensity)
        return jnp.sum(res.res.de_ri)

    val = 1000.
    
    # Initialize 'period' in various shapes to test robustness
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
        
    # --- Automatic Differentiation (AD) ---
    # jax.grad returns a function that computes the gradient of loss_fn w.r.t its first argument.
    grad_fn = jax.grad(loss_fn)
    
    try:
        ad_grad = grad_fn(p_init)
    except Exception as e:
        print(f"    FAILED AD: {e}")
        return

    # --- Finite Difference (FD) ---
    eps = 1e-4
    
    # Helper to compute loss without gradient tracking (standard execution)
    def compute_loss_val(p_in):
        return loss_fn(p_in)

    fd_grad = None

    if shape_type == 'scalar':
         # Central difference for scalar
         l_plus = compute_loss_val(p_init + eps)
         l_minus = compute_loss_val(p_init - eps)
         fd_grad = (l_plus - l_minus) / (2 * eps)
         
    elif shape_type.startswith('vector'):
         # Central difference for each element of the vector
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
         # Central difference for each element of the list
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

    # --- Comparison ---
    if isinstance(ad_grad, list):
        ad_arr = np.array(ad_grad)
        fd_arr = np.array(fd_grad)
    else:
        ad_arr = np.array(ad_grad)
        fd_arr = np.array(fd_grad)

    diff = np.abs(ad_arr - fd_arr).max()
    
    # Check if difference is within tolerance
    if diff < 1e-4:
        print(f"    PASSED (Diff: {diff:.2e})")
        print(f"    AD: {ad_grad}")
        print(f"    FD: {fd_grad}")
    else:
        print(f"    FAILED (Diff: {diff:.2e})")
        print(f"    AD: {ad_grad}")
        print(f"    FD: {fd_grad}")

def run_torch(mode, shape_type):
    """
    Runs gradient check for PyTorch backend.
    """
    print(f"  Testing Torch, mode={mode}, shape={shape_type}")
    
    ucell = get_ucell(mode, 2)
    
    val = 1000.
    
    # Initialize 'period' with requires_grad=True to enable AD
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
        # Set Fourier orders
        fto = [5, 0] if mode == '1D' else [5, 5]
        
        # Call meent
        mee = meent.call_mee(backend=2, period=p_in, ucell=ucell,
                             wavelength=900., thickness=[500.], fto=fto,
                             n_top=1., n_bot=1.5)
        res = mee.conv_solve()
        return res.res.de_ri.sum()

    # --- Automatic Differentiation (AD) ---
    try:
        loss = loss_fn(p)
        loss.backward() # Computes gradients and stores them in p.grad
    except Exception as e:
        print(f"    FAILED AD: {e}")
        return
    
    # Extract gradient values
    if isinstance(p, list):
        ad_grad = [pi.grad.item() for pi in p]
    else:
        ad_grad = p.grad.detach().numpy()
        
    # --- Finite Difference (FD) ---
    eps = 1e-4
    
    # Helper for FD
    def compute_loss_val_torch(p_vals):
        # p_vals contains raw values (floats/arrays).
        # We assume they are effectively constant for the purpose of loss evaluation here.
        if shape_type == 'scalar':
             return loss_fn(torch.tensor(p_vals, dtype=torch.float64))
        elif shape_type.startswith('vector'):
             return loss_fn(torch.tensor(p_vals, dtype=torch.float64))
        elif shape_type.startswith('list'):
             # pass list of tensors
             return loss_fn([torch.tensor(v, dtype=torch.float64) for v in p_vals])
    
    fd_grad = None
    
    with torch.no_grad(): # Disable grad for FD steps
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

    # --- Compare ---
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