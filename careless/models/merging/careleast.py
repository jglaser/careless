import tensorflow as tf
import tf_keras as tfk
import numpy as np
import reciprocalspaceship as rs
import gemmi
from careless.models.base import BaseModel
from tqdm.autonotebook import tqdm

def build_symmetry_map_gemmi(unit_cell_params, space_group_symbol, grid_size):
    """
    Builds a full symmetry expansion map (ASU -> P1 Grid).
    Uses Target Index (h_new) and Negative Phase Shift (-2pi) for correct density reconstruction.
    """
    print(f"Building rigorous symmetry map for {space_group_symbol}...")
    nx, ny, nz_half = grid_size
    
    # 1. Setup Spacegroup and Cell
    sg = gemmi.SpaceGroup(space_group_symbol)
    ops = sg.operations()
    
    if hasattr(unit_cell_params, 'a'): 
        cell = unit_cell_params
    else:
        cell = gemmi.UnitCell(*unit_cell_params)
    
    # 2. Generate P1 Grid Indices
    print("  Generating P1 grid indices...")
    H, K, L = np.meshgrid(
        np.fft.fftfreq(nx, 1/nx).astype(int),
        np.fft.fftfreq(ny, 1/ny).astype(int),
        np.arange(nz_half),
        indexing='ij'
    )
    h_flat, k_flat, l_flat = H.flatten(), K.flatten(), L.flatten()
    
    # Map to ASU
    ds = rs.DataSet({
        'H': h_flat, 'K': k_flat, 'L': l_flat
    }, cell=cell, spacegroup=sg)
    ds.hkl_to_asu(inplace=True)
    
    # Get Unique Indices
    print("  Finding unique ASU parameters...")
    unique_hkls = ds.groupby(['H', 'K', 'L']).first().reset_index()[['H', 'K', 'L']].to_numpy(dtype=np.int32)
    
    # 3. Filter Systematic Absences
    print(f"  Filtering systematic absences from {len(unique_hkls)} unique reflections...")
    keep_mask = np.ones(len(unique_hkls), dtype=bool)
    
    for i in range(len(unique_hkls)):
        h, k, l = unique_hkls[i]
        # Robust absence check
        if ops.is_systematically_absent([int(h), int(k), int(l)]):
            keep_mask[i] = False
            
    indices_asu = unique_hkls[keep_mask]
    n_unique = len(indices_asu)
    print(f"  Unique Reflections (non-absent): {n_unique}")
    
    # 4. Prepare Grid Maps
    total_grid_points = nx * ny * nz_half
    grid_gather_indices = np.full(total_grid_points, -1, dtype=np.int32)
    grid_phase_shifts = np.zeros(total_grid_points, dtype=np.complex64)
    grid_conj_flags = np.zeros(total_grid_points, dtype=bool)
    
    # 5. Iterate Symmetry Operations
    print(f"  Expanding {len(ops)} symmetry operations...")
    
    stride_h = ny * nz_half
    stride_k = nz_half
    stride_l = 1
    
    for op in ops:
        rot = np.array(op.rot, dtype=np.int32).reshape(3, 3)
        trans = np.array(op.tran, dtype=np.float32)
        
        # h_new = h_asu * R
        h_new = np.matmul(indices_asu, rot)
        
        # Phase Shift: -2pi * h_target . t
        # This ensures rho(Rx+t) = rho(x) under TF's +2pi*i exponent convention
        phase_arg = -2.0 * np.pi * np.matmul(h_new, trans)
        phase_shifts = np.exp(1j * phase_arg).astype(np.complex64)
        
        # Friedel Mates (l < 0)
        l_vec = h_new[:, 2]
        is_lower = l_vec < 0
        
        h_final = h_new.copy()
        h_final[is_lower] *= -1
        
        final_shifts = phase_shifts.copy()
        final_shifts[is_lower] = np.conj(final_shifts[is_lower])
        
        # Map to Grid
        h = h_final[:, 0] % nx
        k = h_final[:, 1] % ny
        l = h_final[:, 2]
        
        valid_mask = (l < nz_half)
        
        flat_indices = (h * stride_h + k * stride_k + l).astype(np.int32)
        flat_indices = flat_indices[valid_mask]
        
        current_asu_indices = np.arange(n_unique)[valid_mask]
        current_shifts = final_shifts[valid_mask]
        current_is_lower = is_lower[valid_mask]
        
        grid_gather_indices[flat_indices] = current_asu_indices
        grid_phase_shifts[flat_indices] = current_shifts
        grid_conj_flags[flat_indices] = current_is_lower

    # 6. Handle Absences
    absent_mask = (grid_gather_indices == -1)
    if np.sum(absent_mask) > 0:
        grid_gather_indices[absent_mask] = 0
        grid_phase_shifts[absent_mask] = 0.0 + 0.0j
    
    return grid_gather_indices, grid_phase_shifts, grid_conj_flags, n_unique

class CareleastBase(tfk.models.Model, BaseModel):
    """Base class for Stochastic Phasing Engines"""
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
            def step_fn(model_and_data):
                model, d = model_and_data
                return model.train_step_with_gradient_norm((d,))
        else:
            def step_fn(model_and_data):
                model, d = model_and_data
                return model.train_step((d,))

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
            
            if validation_data is not None:
                if i % validation_frequency == 0:
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
    """
    Spectral SGLD using HALF-GRID representation.
    """
    def __init__(self, asu_collection, likelihood, scaling_model, grid_size, unit_cell, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=True, tv_weight=0.0, prior_weight=0.1, enforce_symmetry=True, stochastic_points=4096, sparsity_weight=0.0, initial_F=None, space_group_symbol=None):
        super().__init__(asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight)
        
        self.grid_size = grid_size
        self.nx, self.ny, self.nz = grid_size
        self.nz_half = self.nz // 2 + 1
        self.n_grid_flat = self.nx * self.ny * self.nz_half
        
        self.use_positivity = use_positivity
        self.tv_weight = tv_weight
        self.stochastic_points = stochastic_points
        self.sparsity_weight = sparsity_weight
        
        self.wilson_sigma_grid = self._precompute_wilson_sigma_grid(unit_cell, b_factor)
       
        # 1. Build the Rigorous Map
        gather_ids, phase_shifts, conj_flags, n_unique = build_symmetry_map_gemmi(
            unit_cell, space_group_symbol, (self.nx, self.ny, self.nz_half)
        )

        self.asu_to_grid_indices = tf.constant(gather_ids, dtype=tf.int32)
        self.asu_phase_shifts = tf.constant(phase_shifts, dtype=tf.complex64)
        self.asu_conj_flags = tf.constant(conj_flags, dtype=tf.bool)
        self.n_unique = n_unique

        # 2. Initialize Unique Parameters
        print(f"Initializing F_state with {n_unique} unique parameters (was {self.n_grid_flat}).")

        init_real = tf.random.normal((self.n_particles, n_unique), stddev=0.1)
        init_imag = tf.random.normal((self.n_particles, n_unique), stddev=0.1)
        init_val = tf.complex(init_real, init_imag)

        self.F_state = tf.Variable(init_val, name='F_state', dtype=tf.complex64)
        self.momentum = tf.Variable(tf.zeros_like(self.F_state), trainable=False, name='momentum', dtype=tf.complex64)

        if initial_F is not None:
             # Just random init for now to be safe, handling complex gathered init is tricky
             pass

        self.gather_indices, self.gather_friedel_mask = self._precompute_gather_indices()
        
        # We are using Full Grid FFT (Dense mode)
        self.sparse_mode = False 

    def _precompute_wilson_sigma_grid(self, unit_cell, b_factor):
        uc = unit_cell.parameters
        kx = np.fft.fftfreq(self.nx).astype(np.float32) * self.nx
        ky = np.fft.fftfreq(self.ny).astype(np.float32) * self.ny
        kz = np.fft.rfftfreq(self.nz).astype(np.float32) * self.nz 
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        s_sq = (KX / uc[0])**2 + (KY / uc[1])**2 + (KZ / uc[2])**2
        sigma = np.exp(-b_factor * s_sq / 2.0).astype(np.float32)
        sigma /= sigma.max()
        return tf.constant(np.maximum(sigma, 1e-6))

    def _precompute_gather_indices(self):
        lookup = self.asu_collection.reciprocal_asus[0].lookup_table
        hkls = lookup.get_hkls().astype(np.int32)
        h = np.mod(hkls[:, 0], self.nx)
        k = np.mod(hkls[:, 1], self.ny)
        l = np.mod(hkls[:, 2], self.nz)
        
        is_friedel = l > (self.nz // 2)
        idx_h = np.where(is_friedel, (self.nx - h) % self.nx, h)
        idx_k = np.where(is_friedel, (self.ny - k) % self.ny, k)
        idx_l = np.where(is_friedel, (self.nz - l) % self.nz, l)
        
        indices = np.stack([idx_h, idx_k, idx_l], axis=1)
        return tf.constant(indices, dtype=tf.int32), tf.constant(is_friedel, dtype=tf.bool)

    def _gather_F_sparse(self, F_flat_state):
        flat_indices = (self.gather_indices[:, 0] * self.ny * self.nz_half) + \
                       (self.gather_indices[:, 1] * self.nz_half) + \
                       self.gather_indices[:, 2]
        F_vals = tf.gather(F_flat_state, flat_indices, axis=1)
        return F_vals, flat_indices 
    
    def _apply_friedel(self, F_vals):
        mask = self.gather_friedel_mask[None, :]
        return tf.where(mask, tf.math.conj(F_vals), F_vals)

    def _expand_to_full(self, F_flat_state):
        F_half = tf.reshape(F_flat_state, (self.n_particles, self.nx, self.ny, self.nz_half))
        limit = -1 if (self.nz % 2 == 0) else None
        middle_slice = F_half[..., 1:limit]
        rev_slice = tf.reverse(middle_slice, axis=[1, 2, 3])
        rev_slice = tf.roll(rev_slice, shift=[1, 1, 0], axis=[1, 2, 3])
        redundant_part = tf.math.conj(rev_slice)
        return tf.concat([F_half, redundant_part], axis=-1)

    def _nan_safe_complex(self, x):
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        is_finite = tf.logical_and(tf.math.is_finite(real), tf.math.is_finite(imag))
        real = tf.where(is_finite, real, tf.zeros_like(real))
        imag = tf.where(is_finite, imag, tf.zeros_like(imag))
        return tf.complex(real, imag)
    
    def _nan_safe_real(self, x):
        return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

    def call(self, inputs):
        # Forward pass for evaluation
        # Gather with Symmetry Expansion
        F_gathered = tf.gather(self.F_state, self.asu_to_grid_indices, axis=1)
        F_expanded = tf.where(self.asu_conj_flags, tf.math.conj(F_gathered), F_gathered)
        F_grid_flat = F_expanded * self.asu_phase_shifts

        # Map to HKL
        flat_indices = (self.gather_indices[:, 0] * self.ny * self.nz_half) + \
                       (self.gather_indices[:, 1] * self.nz_half) + \
                       self.gather_indices[:, 2]
        F_vals = tf.gather(F_grid_flat, flat_indices, axis=1)
        
        refl_id = tf.squeeze(self.get_refl_id(inputs), axis=-1)
        F_obs_complex = tf.gather(self._apply_friedel(F_vals), refl_id, axis=1)
        F_abs_sq = tf.square(tf.abs(F_obs_complex))
        
        scale_dist = self.scaling_model(inputs)
        z_scale = scale_dist.sample(self.n_particles)
        if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
        z_scale = tf.where(tf.math.is_finite(z_scale), z_scale, tf.ones_like(z_scale))
        return z_scale * F_abs_sq

    def train_step(self, data, train_structure=True, train_scale=True, fix_scale_unity=False):
        # 1. Unpack Data & Batch Size
        inputs = data[0]
        batch_size = tf.cast(tf.shape(inputs[0])[0], tf.float32)

        # Extract Indices
        h_in = tf.cast(inputs[0], tf.int32)
        k_in = tf.cast(inputs[1], tf.int32)
        l_in = tf.cast(inputs[2], tf.int32)
        
        h_in = tf.reshape(h_in, [-1])
        k_in = tf.reshape(k_in, [-1])
        l_in = tf.reshape(l_in, [-1])

        with tf.GradientTape() as tape:
            # --- 1. EXPANSION (ASU -> P1) ---
            # A. Gather Unique Parameters
            F_gathered = tf.gather(self.F_state, self.asu_to_grid_indices, axis=1)
            
            # B. Apply Conjugation (Friedel Mates)
            F_expanded = tf.where(
                self.asu_conj_flags, 
                tf.math.conj(F_gathered), 
                F_gathered
            )
            
            # C. Apply Phase Shifts (Symmetry Translation) & Filter Absences
            F_grid_flat = F_expanded * self.asu_phase_shifts
            
            # --- 2. DENSITY ---
            F_grid_half = tf.reshape(F_grid_flat, (self.n_particles, self.nx, self.ny, self.nz_half))
            F_full = self._expand_to_full(F_grid_half)
            
            # [FIX 1] Density Scaling (Normalize IFFT to Electrons)
            total_grid_points = tf.cast(self.nx * self.ny * (2 * self.nz_half - 2), tf.float32)
            rho = tf.math.real(tf.signal.ifft3d(F_full)) * total_grid_points
            
            # --- 3. DYNAMIC GATHER ---
            is_friedel = l_in < 0
            h = tf.where(is_friedel, -h_in, h_in)
            k = tf.where(is_friedel, -k_in, k_in)
            l = tf.where(is_friedel, -l_in, l_in)
            h = h % self.nx
            k = k % self.ny
            flat_indices = (h * self.ny * self.nz_half) + (k * self.nz_half) + l
            
            F_batch = tf.gather(F_grid_flat, flat_indices, axis=1)
            F_abs_sq = tf.square(tf.abs(F_batch))
            
            # --- 4. SCALING ---
            if fix_scale_unity:
                z_scale = tf.ones((1, tf.shape(F_abs_sq)[1]), dtype=tf.float32)
            else:
                z_dist = self.scaling_model(inputs)
                z_sample = z_dist.sample()
                z_scale = tf.reshape(z_sample, (1, -1))

            ipred = z_scale * F_abs_sq
            ipred = tf.clip_by_value(ipred, 1e-12, 1e15)

            # --- 5. LIKELIHOOD ---
            likelihood = self.likelihood(inputs)
            log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
            nll = -tf.reduce_sum(log_lik)
            
            # --- 6. PRIORS ---
            prior_dist = 0.0
            
            # A. Sparsity (L1 Norm)
            if self.sparsity_weight > 0:
                rho_mean = tf.reduce_mean(tf.abs(rho), axis=0)
                sparsity_term = tf.reduce_mean(rho_mean) 
                prior_dist += self.sparsity_weight * sparsity_term
                
            # B. Positivity (X-ray)
            if self.use_positivity:
                rho_mean_real = tf.reduce_mean(rho, axis=0)
                pos_term = 10.0 * tf.reduce_mean(tf.square(tf.nn.relu(-rho_mean_real)))
                prior_dist += pos_term

            # [FIX 2] Total Variation (TV)
            if self.tv_weight > 0.0:
                eps = 1e-6
                dx = rho - tf.roll(rho, shift=1, axis=1)
                dy = rho - tf.roll(rho, shift=1, axis=2)
                dz = rho - tf.roll(rho, shift=1, axis=3)
                grad_mag = tf.sqrt(tf.square(dx) + tf.square(dy) + tf.square(dz) + eps)
                tv_term = tf.reduce_mean(grad_mag)
                prior_dist += self.tv_weight * tv_term

            # [FIX 3] Batch Scaling
            prior_energy = prior_dist * batch_size
            
            loss = nll + prior_energy

        # --- 7. GRADIENTS ---
        vars_structure = [self.F_state]
        vars_scale = self.scaling_model.trainable_variables
        all_vars = vars_structure + vars_scale
        
        grads = tape.gradient(loss, all_vars)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, all_vars))
        
        # --- STATS FOR MONITORING ---
        rho_avg = tf.reduce_mean(rho, axis=0)
        rho_flat = tf.reshape(rho_avg, [-1])
        mean, var = tf.nn.moments(rho_flat, axes=[0])
        std = tf.sqrt(var + 1e-8)
        skewness = tf.reduce_mean(tf.pow(rho_flat - mean, 3)) / tf.pow(std, 3)
        kurtosis = tf.reduce_mean(tf.pow(rho_flat - mean, 4)) / tf.pow(std, 4)
        max_rho = tf.reduce_max(rho_flat)
        grad_norm = tf.norm(grads[0]) if grads[0] is not None else 0.0

        return {
            "loss": loss,
            "nll": nll,
            "prior": prior_energy,
            "grad_norm": grad_norm,
            "Skew": skewness,
            "Kurt": kurtosis,
            "Max": max_rho
        }

    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        ipred = self(data)
        likelihood = self.likelihood(data)
        log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
        return {"loss": -tf.reduce_mean(log_lik), "NLL": -tf.reduce_mean(log_lik)}


class CareleastRealSpace(CareleastBase):
    # This class remains as provided in your file for reference/completeness
    def __init__(self, asu_collection, likelihood, scaling_model, grid_size, unit_cell, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=True, tv_weight=0.0, prior_weight=0.1):
        super().__init__(asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight)
        self.grid_size = grid_size
        self.use_positivity = use_positivity
        self.tv_weight = tv_weight
        self.n_pixels = tf.cast(tf.reduce_prod(grid_size), tf.float32)
        self.hkl_to_grid_indices = self._precompute_grid_indices()
        self.wilson_spectrum = self._precompute_wilson_spectrum(unit_cell, b_factor)
        self.density = tf.Variable(tf.random.normal((n_particles, *grid_size)) * 0.001, name='density', dtype=tf.float32)
        self.momentum = tf.Variable(tf.zeros_like(self.density), trainable=False, name='momentum')

    def _precompute_grid_indices(self):
        lookup = self.asu_collection.reciprocal_asus[0].lookup_table
        hkls = lookup.get_hkls().astype(np.int32)
        nx, ny, nz = self.grid_size
        h = np.mod(hkls[:, 0], nx)
        k = np.mod(hkls[:, 1], ny)
        l = np.mod(hkls[:, 2], nz)
        return tf.stack([h, k, l], axis=1)

    def _precompute_wilson_spectrum(self, unit_cell, b_factor):
        nx, ny, nz = self.grid_size
        uc = unit_cell.parameters 
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        kz = np.fft.fftfreq(nz) * nz
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        s_sq = (KX / uc[0])**2 + (KY / uc[1])**2 + (KZ / uc[2])**2
        spectrum = np.exp(-b_factor * s_sq / 2.0).astype(np.float32)
        spectrum /= spectrum.max()
        return tf.constant(np.maximum(spectrum, 1e-6), dtype=tf.complex64)

    def _fft_forward(self, density):
        d_complex = tf.complex(density, tf.zeros_like(density))
        return tf.signal.fft3d(d_complex)

    def _ifft_backward(self, F_grid):
        return tf.math.real(tf.signal.ifft3d(F_grid))
    
    def _nan_safe_real(self, x):
        return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

    def call(self, inputs):
        F_grid = self._fft_forward(self.density)
        refl_id = tf.squeeze(self.get_refl_id(inputs), axis=-1)
        grid_indices = tf.gather(self.hkl_to_grid_indices, refl_id)
        nx, ny, nz = self.grid_size
        flat_indices = (grid_indices[:, 0] * ny * nz) + (grid_indices[:, 1] * nz) + grid_indices[:, 2]
        F_flat = tf.reshape(F_grid, (self.n_particles, -1))
        F_obs = tf.gather(F_flat, flat_indices, axis=1)
        scale_dist = self.scaling_model(inputs)
        z_scale = scale_dist.sample(self.n_particles)
        if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
        z_scale = tf.where(tf.math.is_finite(z_scale), z_scale, tf.ones_like(z_scale))
        return z_scale * tf.square(tf.abs(F_obs))

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        with tf.GradientTape() as tape:
            tape.watch(self.density)
            scale_dist = self.scaling_model(data)
            z_scale = scale_dist.sample(self.n_particles)
            if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
            z_scale = tf.where(tf.math.is_finite(z_scale), z_scale, tf.ones_like(z_scale))
            F_grid = self._fft_forward(self.density)
            refl_id = tf.squeeze(self.get_refl_id(data), axis=-1)
            grid_indices = tf.gather(self.hkl_to_grid_indices, refl_id)
            nx, ny, nz = self.grid_size
            flat_indices = (grid_indices[:, 0] * ny * nz) + (grid_indices[:, 1] * nz) + grid_indices[:, 2]
            F_flat = tf.reshape(F_grid, (self.n_particles, -1))
            F_obs = tf.gather(F_flat, flat_indices, axis=1)
            ipred = z_scale * tf.square(tf.abs(F_obs))
            ipred = tf.clip_by_value(ipred, 1e-12, 1e15)
            likelihood = self.likelihood(data)
            log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
            F_sq = tf.square(tf.abs(F_grid))
            wilson_energy = 0.5 * tf.reduce_mean(F_sq / tf.abs(self.wilson_spectrum), axis=[1,2,3])
            prior_energy = self.prior_weight * wilson_energy
            if self.use_positivity:
                prior_energy += 10.0 * tf.reduce_mean(tf.square(tf.nn.relu(-self.density)), axis=[1,2,3])
            if self.tv_weight > 0.0:
                 eps = 1e-6
                 dx = tf.sqrt(tf.square(self.density - tf.roll(self.density, 1, axis=1)) + eps)
                 dy = tf.sqrt(tf.square(self.density - tf.roll(self.density, 1, axis=2)) + eps)
                 dz = tf.sqrt(tf.square(self.density - tf.roll(self.density, 1, axis=3)) + eps)
                 prior_energy += self.tv_weight * tf.reduce_mean(dx + dy + dz, axis=[1,2,3])
            loss = -tf.reduce_mean(log_lik) + tf.reduce_mean(prior_energy)
        grads = tape.gradient(loss, [self.density] + self.scaling_model.trainable_variables)
        grad_rho = grads[0]
        safe_grads_scaling = []
        if len(grads) > 1:
            for g in grads[1:]:
                if g is not None: safe_grads_scaling.append(self._nan_safe_real(g))
                else: safe_grads_scaling.append(None)
        grad_rho = self._nan_safe_real(grad_rho)
        grad_rho = tf.clip_by_global_norm([grad_rho], 10.0)[0][0]
        grads_k = self._fft_forward(grad_rho) * self.wilson_spectrum
        grads_smooth = self._ifft_backward(grads_k)
        grads_smooth = tf.clip_by_global_norm([grads_smooth], 100.0)[0][0]
        noise = tf.random.normal(tf.shape(self.density)) * tf.sqrt(2.0 * self.learning_rate * self.temperatures)[:, None, None, None]
        noise_colored = self._ifft_backward(self._fft_forward(noise) * tf.sqrt(self.wilson_spectrum))
        new_momentum = (self.momentum * self.friction) - (self.learning_rate * grads_smooth) + noise_colored
        self.density.assign_add(new_momentum)
        self.momentum.assign(new_momentum)
        if safe_grads_scaling:
            self.optimizer.apply_gradients(zip(safe_grads_scaling, self.scaling_model.trainable_variables))
        return {"loss": loss, "lik": tf.reduce_mean(log_lik)}
