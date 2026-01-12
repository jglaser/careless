import tensorflow as tf
import tf_keras as tfk
import numpy as np
import reciprocalspaceship as rs
import gemmi
from careless.models.base import BaseModel
from tqdm.autonotebook import tqdm

# --- 1. ROBUST NORM HELPER (Fixes NaN in clipping) ---
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
    Wraps tf.signal.ifft3d.
    Backward pass prints the Gradient Norm to confirm connectivity.
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
        
        # 4. Print Probe (The "Single Probe" requested)
        # [FIX] Removed f-string formatting for tensors. Passed as separate args.
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
    """
    print(f"Building rigorous symmetry map for {space_group_symbol}...")
    nx, ny, nz_half = grid_size
    
    sg = gemmi.SpaceGroup(space_group_symbol)
    ops = sg.operations()
    
    if hasattr(unit_cell_params, 'a'): 
        cell = unit_cell_params
    else:
        cell = gemmi.UnitCell(*unit_cell_params)
    
    # Generate P1 Grid Indices
    H, K, L = np.meshgrid(
        np.fft.fftfreq(nx, 1/nx).astype(int),
        np.fft.fftfreq(ny, 1/ny).astype(int),
        np.arange(nz_half),
        indexing='ij'
    )
    h_flat, k_flat, l_flat = H.flatten(), K.flatten(), L.flatten()
    
    ds = rs.DataSet({'H': h_flat, 'K': k_flat, 'L': l_flat}, cell=cell, spacegroup=sg)
    ds.hkl_to_asu(inplace=True)
    
    unique_hkls = ds.groupby(['H', 'K', 'L']).first().reset_index()[['H', 'K', 'L']].to_numpy(dtype=np.int32)
    
    # Filter Absences
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
    
    for op in ops:
        den = float(op.DEN)
        rot = np.array(op.rot, dtype=np.float32).reshape(3, 3) / den
        trans = np.array(op.tran, dtype=np.float32) / den

        h_new = np.rint(np.matmul(indices_asu.astype(np.float32), rot)).astype(np.int32)
        phase_arg = 2.0 * np.pi * np.matmul(indices_asu.astype(np.float32), trans)
        phase_shifts = np.exp(1j * phase_arg).astype(np.complex64)

        l_vec = h_new[:, 2]
        is_lower = l_vec < 0

        h_final = h_new.copy()
        h_final[is_lower] *= -1
        
        final_shifts = phase_shifts.copy()
        final_shifts[is_lower] = np.conj(final_shifts[is_lower])

        h = h_final[:, 0] % nx
        k = h_final[:, 1] % ny
        l = h_final[:, 2]

        valid_mask = (l < nz_half)
        flat_indices = (h * stride_h + k * stride_k + l).astype(np.int32)[valid_mask]

        grid_gather_indices[flat_indices] = np.arange(n_unique)[valid_mask]
        grid_phase_shifts[flat_indices] = final_shifts[valid_mask]
        grid_conj_flags[flat_indices] = is_lower[valid_mask]

    # Handle unmapped (absent) regions
    absent_mask = (grid_gather_indices == -1)
    if np.sum(absent_mask) > 0:
        grid_gather_indices[absent_mask] = 0
        grid_phase_shifts[absent_mask] = 0.0

    return grid_gather_indices, grid_phase_shifts, grid_conj_flags, n_unique

class CareleastBase(tfk.models.Model, BaseModel):
    def __init__(self, asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight=0.1):
        super().__init__()
        self.asu_collection = asu_collection
        self.likelihood = likelihood
        self.scaling_model = scaling_model
        self.n_particles = n_particles
        self.learning_rate = learning_rate
        self.friction = friction
        self.prior_weight = prior_weight
        
        if temperatures is None:
            temperatures = np.geomspace(1.0, 2.5, n_particles)
        else:
            temperatures = np.array(temperatures, dtype=np.float32)
        self.temperatures = tf.Variable(temperatures, trainable=False, dtype=tf.float32, name="temperatures")

    def train_step_with_gradient_norm(self, data=None):
        if data is None: x = self.data 
        elif isinstance(data, tuple): x = data[0]
        else: x = data
        metrics = self.train_step((x,))
        if "Grad Norm" not in metrics: metrics["Grad Norm"] = 0.0
        return metrics

    def train_model(self, data, steps, message=None, format_string="{:0.2e}", validation_data=None, validation_frequency=10, progress=True, use_custom_train_step=True, jit_compile=None, reduce_retracing=False):
        if use_custom_train_step:
            step_fn = lambda d: self.train_step_with_gradient_norm((d[1],))
        else:
            step_fn = lambda d: self.train_step((d[1],))

        if not self._run_eagerly:
            step_fn = tf.function(step_fn, reduce_retracing=reduce_retracing, jit_compile=jit_compile)

        if validation_data is not None:
             val_scale = len(data[0]) / len(validation_data[0])

        history = {}
        from tqdm import trange
        disable_progress = not progress
        bar = trange(steps, desc=message, disable=disable_progress)
        
        for i in bar:
            _history = step_fn((self, data))
            
            if validation_data is not None and i % validation_frequency == 0:
                validation_metrics = self.test_on_batch(validation_data, return_dict=True)
                _history['NLL_val'] = val_scale * validation_metrics['NLL']

            pf = {}
            for k,v in _history.items():
                v = float(v)
                pf[k] = format_string.format(v)
                if k not in history: history[k] = []
                history[k].append(v)

            bar.set_postfix(pf)
        return history

class CareleastSpectral(CareleastBase):
    def __init__(self, asu_collection, likelihood, scaling_model, grid_size, unit_cell, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=True, tv_weight=0.0, prior_weight=0.1, enforce_symmetry=True, stochastic_points=4096, sparsity_weight=0.0, initial_F=None, space_group_symbol=None):
        super().__init__(asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight)
        
        self.grid_size = grid_size
        self.nx, self.ny, self.nz = grid_size
        self.nz_half = self.nz // 2 + 1
        
        self.use_positivity = use_positivity
        self.tv_weight = tv_weight
        self.sparsity_weight = sparsity_weight
        
        # Build Map
        gather_ids, phase_shifts, conj_flags, n_unique = build_symmetry_map_gemmi(
            unit_cell, space_group_symbol, (self.nx, self.ny, self.nz_half)
        )

        self.asu_to_grid_indices = tf.constant(gather_ids, dtype=tf.int32)
        self.asu_phase_shifts = tf.constant(phase_shifts, dtype=tf.complex64)
        self.asu_conj_flags = tf.constant(conj_flags, dtype=tf.bool)
        self.n_unique = n_unique

        # Init Parameters
        init_val = tf.complex(
            tf.random.normal((self.n_particles, n_unique), stddev=0.1),
            tf.random.normal((self.n_particles, n_unique), stddev=0.1)
        )
        self.F_state = tf.Variable(init_val, name='F_state', dtype=tf.complex64)
        
    def _expand_to_full(self, F_flat_state):
        F_half = tf.reshape(F_flat_state, (self.n_particles, self.nx, self.ny, self.nz_half))
        limit = -1 if (self.nz % 2 == 0) else None
        
        # Hermitian Symmetry Logic for Real-Space Density
        middle_slice = F_half[..., 1:limit]
        rev_slice = tf.reverse(middle_slice, axis=[1, 2, 3])
        rev_slice = tf.roll(rev_slice, shift=[1, 1, 0], axis=[1, 2, 3])
        redundant_part = tf.math.conj(rev_slice)
        
        return tf.concat([F_half, redundant_part], axis=-1)

    def train_step(self, data, train_structure=True, train_scale=True, fix_scale_unity=False):
        inputs = data[0]
        batch_size = tf.cast(tf.shape(inputs[0])[0], tf.float32)

        # Extract Indices
        h_in = tf.cast(inputs[0], tf.int32)
        k_in = tf.cast(inputs[1], tf.int32)
        l_in = tf.cast(inputs[2], tf.int32)
        h_in, k_in, l_in = [tf.reshape(x, [-1]) for x in (h_in, k_in, l_in)]

        with tf.GradientTape() as tape:
            # --- 1. DENSITY CALCULATION (With Probe) ---
            # A. Expand ASU Parameters to Half-Grid
            F_gathered = tf.gather(self.F_state, self.asu_to_grid_indices, axis=1)
            F_expanded = tf.where(self.asu_conj_flags, tf.math.conj(F_gathered), F_gathered)
            F_grid_flat = F_expanded * self.asu_phase_shifts
            
            # B. Expand Half-Grid to Full-Grid (Hermitian)
            F_grid_half = tf.reshape(F_grid_flat, (self.n_particles, self.nx, self.ny, self.nz_half))
            F_full = self._expand_to_full(F_grid_half)
            
            # C. IFFT (Wrapped in Probe to ensure Gradient Flow)
            rho_complex = probed_ifft(F_full)
            
            # D. Scaling (Matches TF's 1/N normalization in IFFT)
            total_grid_points = tf.cast(self.nx * self.ny * (2 * self.nz_half - 2), tf.float32)
            rho = tf.math.real(rho_complex) * total_grid_points
            
            # --- 2. LIKELIHOOD ---
            # Dynamic Gathering for Sparse Data
            is_friedel = l_in < 0
            h = tf.where(is_friedel, -h_in, h_in) % self.nx
            k = tf.where(is_friedel, -k_in, k_in) % self.ny
            l = tf.where(is_friedel, -l_in, l_in)
            
            flat_indices = (h * self.ny * self.nz_half) + (k * self.nz_half) + l
            F_batch = tf.gather(F_grid_flat, flat_indices, axis=1)
            F_abs_sq = tf.square(tf.abs(F_batch))
            
            # Scaling Model
            z_scale = self.scaling_model(inputs).sample()
            z_scale = tf.reshape(z_scale, (1, -1))
            ipred = tf.clip_by_value(z_scale * F_abs_sq, 1e-12, 1e15)

            # NLL
            likelihood = self.likelihood(inputs)
            nll = -tf.reduce_sum(tf.reduce_sum(likelihood.log_prob(ipred), axis=-1))
            
            # --- 3. PRIORS ---
            prior_dist = 0.0
            
            # Sparsity (L1)
            if self.sparsity_weight > 0:
                prior_dist += self.sparsity_weight * tf.reduce_mean(tf.abs(rho))
                
            # Positivity (ReLU)
            if self.use_positivity:
                prior_dist += 0.1 * tf.reduce_mean(tf.square(tf.nn.relu(-rho)))

            # Total Variation
            if self.tv_weight > 0.0:
                eps = 1e-6
                dx = rho - tf.roll(rho, shift=1, axis=1)
                dy = rho - tf.roll(rho, shift=1, axis=2)
                dz = rho - tf.roll(rho, shift=1, axis=3)
                grad_mag = tf.sqrt(tf.square(dx) + tf.square(dy) + tf.square(dz) + eps)
                prior_dist += self.tv_weight * tf.reduce_mean(grad_mag)

            prior_energy = prior_dist * batch_size
            loss = nll + prior_energy

        # --- 4. GRADIENTS & UPDATES (Manual Clipping) ---
        all_vars = [self.F_state] + self.scaling_model.trainable_variables
        grads = tape.gradient(loss, all_vars)

        # Manual Global Norm Calculation (Float32) to avoid Complex64 NaN
        grads_real_parts = []
        for g in grads:
            if g is not None:
                if g.dtype.is_complex:
                    grads_real_parts.extend([tf.math.real(g), tf.math.imag(g)])
                else:
                    grads_real_parts.append(g)
        
        squared_sums = [tf.reduce_sum(tf.square(g)) for g in grads_real_parts]
        global_norm = tf.sqrt(tf.reduce_sum(tf.stack(squared_sums)) + 1e-8)
        
        # Apply Clipping
        clip_norm = 10.0
        scale = tf.minimum(1.0, clip_norm / (global_norm + 1e-8))
        
        grads_final = []
        for g in grads:
            if g is not None:
                if g.dtype.is_complex:
                    g_scaled = tf.complex(tf.math.real(g) * scale, tf.math.imag(g) * scale)
                    grads_final.append(g_scaled)
                else:
                    grads_final.append(g * scale)
            else:
                grads_final.append(None)

        self.optimizer.apply_gradients(zip(grads_final, all_vars))

        rho_flat = tf.reshape(tf.reduce_mean(rho, axis=0), [-1])
        return {
            "loss": loss,
            "nll": nll,
            "prior": prior_energy,
            "grad_norm": global_norm,
            "Max": tf.reduce_max(rho_flat)
        }

class CareleastRealSpace(CareleastBase):
    pass # (Real space implementation omitted for brevity as Spectral is active)
