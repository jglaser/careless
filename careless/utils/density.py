import tensorflow as tf
import numpy as np
import reciprocalspaceship as rs
import gemmi

# --- 1. ROBUST NORM HELPER ---
def safe_complex_norm(x):
    """
    Computes Euclidean norm of complex tensor using only real math.
    Workaround for broken tf.norm(complex64) on ROCm/AMD.
    """
    real = tf.math.real(x)
    imag = tf.math.imag(x)
    # Sum of squares (Real)
    sq_sum = tf.reduce_sum(tf.square(real)) + tf.reduce_sum(tf.square(imag))
    # Sqrt
    return tf.sqrt(sq_sum + 1e-8)

# --- 2. SINGLE GRADIENT PROBE (Heartbeat Monitor) ---
@tf.custom_gradient
def probed_ifft(x):
    """
    Wraps tf.signal.ifft3d with a gradient probe.
    """
    # Forward Pass
    y = tf.signal.ifft3d(x)
    
    def grad(dy):
        # Backward Pass
        # dy is the gradient arriving from Real-Space constraints
        
        # 1. Check Incoming Health (Real-Space Gradient)
        dy_norm = safe_complex_norm(dy)
        
        # 2. Compute Gradient (Fourier-Space Gradient)
        # grad(IFFT(x)) -> (1/N) * FFT(dy)
        raw_fft = tf.signal.fft3d(dy)
        
        # 3. Check Outgoing Health
        res_norm = safe_complex_norm(raw_fft)
        
        # 4. Print Probe
        # Note: We print periodically or if values are huge to avoid spam if desired
        # For now, print every step to debug the SGLD kickoff
        op = tf.print("[Gradient Probe] Real-Space Grad:", dy_norm, "-> Spectral Grad:", res_norm)
        
        with tf.control_dependencies([op]):
            # Apply N normalization for IFFT gradient
            shape = tf.shape(x)
            N = tf.cast(shape[1]*shape[2]*shape[3], x.dtype)
            final_grad = raw_fft / N
            return final_grad

    return y, grad

def build_symmetry_map_gemmi(unit_cell_params, space_group_symbol, grid_size):
    """
    Builds a full symmetry expansion map (ASU -> P1 Grid).
    Returns gather indices and phase shifts to reconstruct the P1 reciprocal grid 
    from unique ASU reflections.
    """
    print(f"Building rigorous symmetry map for {space_group_symbol}...")
    nx, ny, nz_half = grid_size
    
    sg = gemmi.SpaceGroup(space_group_symbol)
    ops = sg.operations()
    
    if hasattr(unit_cell_params, 'a'): 
        cell = unit_cell_params
    else:
        cell = gemmi.UnitCell(*unit_cell_params)
    
    # Generate P1 Grid Indices (Half-sphere in Z for Hermitian symmetry later)
    H, K, L = np.meshgrid(
        np.fft.fftfreq(nx, 1/nx).astype(int),
        np.fft.fftfreq(ny, 1/ny).astype(int),
        np.arange(nz_half),
        indexing='ij'
    )
    h_flat, k_flat, l_flat = H.flatten(), K.flatten(), L.flatten()
    
    ds = rs.DataSet({'H': h_flat, 'K': k_flat, 'L': l_flat}, cell=cell, spacegroup=sg)
    ds.hkl_to_asu(inplace=True)
    
    # Extract unique HKLs that map to this grid
    unique_hkls = ds.groupby(['H', 'K', 'L']).first().reset_index()[['H', 'K', 'L']].to_numpy(dtype=np.int32)
    
    # Filter Systematic Absences
    keep_mask = np.ones(len(unique_hkls), dtype=bool)
    for i in range(len(unique_hkls)):
        if ops.is_systematically_absent([int(x) for x in unique_hkls[i]]):
            keep_mask[i] = False
            
    indices_asu = unique_hkls[keep_mask]
    n_unique = len(indices_asu)
    print(f"  Unique Reflections (non-absent): {n_unique}")
    
    # Prepare Maps
    total_grid_points = nx * ny * nz_half
    grid_gather_indices = np.full(total_grid_points, -1, dtype=np.int32)
    grid_phase_shifts = np.zeros(total_grid_points, dtype=np.complex64)
    grid_conj_flags = np.zeros(total_grid_points, dtype=bool)
    
    stride_h = ny * nz_half
    stride_k = nz_half
    
    # Map every ASU reflection to all its symmetry equivalents on the grid
    # We loop over operators -> forward mapping
    # (Alternatively, we could loop over grid points -> backward mapping, 
    # but forward mapping ensures we hit the exact operator relation used by Gemmi)
    
    # Actually, the most robust way is:
    # 1. We know the grid HKLs (h_flat, etc.)
    # 2. We mapped them to ASU (ds['H'], etc.)
    # 3. We match the ASU HKLs to our unique list `indices_asu`
    
    # Let's use a lookup for the unique indices
    # Create a structured array for fast matching or a dictionary
    asu_lookup = {tuple(hkl): i for i, hkl in enumerate(indices_asu)}
    
    # Iterate over the grid points
    # This is slow in pure Python, so we use the Gemmi/Pandas way if possible.
    # The previous implementation (looping ops) was vectorized. Let's stick to that if it works.
    
    for op in ops:
        den = float(op.DEN)
        rot = np.array(op.rot, dtype=np.float32).reshape(3, 3) / den
        trans = np.array(op.tran, dtype=np.float32) / den

        # Apply operator to all unique ASU reflections
        h_new = np.rint(np.matmul(indices_asu.astype(np.float32), rot)).astype(np.int32)
        phase_arg = 2.0 * np.pi * np.matmul(indices_asu.astype(np.float32), trans)
        phase_shifts = np.exp(1j * phase_arg).astype(np.complex64)

        # Handle Friedel pairs (if L < 0, invert)
        l_vec = h_new[:, 2]
        is_lower = l_vec < 0

        h_final = h_new.copy()
        h_final[is_lower] *= -1
        
        final_shifts = phase_shifts.copy()
        final_shifts[is_lower] = np.conj(final_shifts[is_lower])

        # Wrap indices to grid
        h = h_final[:, 0] % nx
        k = h_final[:, 1] % ny
        l = h_final[:, 2]

        # Only keep those that fall in the upper hemisphere (L >= 0) defined by our grid
        valid_mask = (l < nz_half)
        
        flat_indices = (h * stride_h + k * stride_k + l).astype(np.int32)[valid_mask]

        # Fill the maps
        # Note: Overwrites happen if special position multiplicities exist. 
        # This effectively handles the epsilon factors implicitly by last-write-wins 
        # or we might need scaling. For phasing, usually phase consistency is enough.
        grid_gather_indices[flat_indices] = np.arange(n_unique)[valid_mask]
        grid_phase_shifts[flat_indices] = final_shifts[valid_mask]
        grid_conj_flags[flat_indices] = is_lower[valid_mask]

    # Handle unmapped regions (Systematic absences on the grid)
    absent_mask = (grid_gather_indices == -1)
    if np.sum(absent_mask) > 0:
        # Map them to index 0 with 0 amplitude to zero them out
        grid_gather_indices[absent_mask] = 0
        grid_phase_shifts[absent_mask] = 0.0

    return grid_gather_indices, grid_phase_shifts, grid_conj_flags, n_unique
