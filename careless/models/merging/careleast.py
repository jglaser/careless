import tensorflow as tf
import tf_keras as tfk
import numpy as np
import reciprocalspaceship as rs
from careless.models.base import BaseModel
from tqdm.autonotebook import tqdm

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
    
    Optimizations:
    1. **Vectorized Sparse Gradients:** Process all particles in parallel.
    2. **Stochastic Real-Space Constraints:** Applies Positivity/Sparsity via random point sampling (DFT).
    """
    def __init__(self, asu_collection, likelihood, scaling_model, grid_size, unit_cell, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=True, tv_weight=0.0, prior_weight=0.1, enforce_symmetry=True, stochastic_points=4096, sparsity_weight=0.0, initial_F=None):
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
       
        # --- MODIFIED INITIALIZATION ---
        if initial_F is not None:
            print("Careleast: Initializing state from provided grid (Simulated Phases).")
            
            # --- FIX: Explicitly cast to complex64 ---
            initial_F = tf.cast(initial_F, tf.complex64)
            
            # initial_F is expected to be shape (nx, ny, nz_half)
            init_F_flat_single = tf.reshape(initial_F, (self.n_grid_flat,))
            init_F_flat = tf.tile(init_F_flat_single[None, :], [n_particles, 1])
        else:
            # Standard Random Initialization
            init_sigma = self.wilson_sigma_grid
            init_std = tf.sqrt(init_sigma / 2.0)
            init_F_real = tf.random.normal((n_particles, self.nx, self.ny, self.nz_half)) * init_std
            init_F_imag = tf.random.normal((n_particles, self.nx, self.ny, self.nz_half)) * init_std
            init_F_flat = tf.reshape(tf.complex(init_F_real, init_F_imag), (n_particles, self.n_grid_flat))

        self.F_state = tf.Variable(init_F_flat, name='F_state', dtype=tf.complex64)
        self.momentum = tf.Variable(tf.zeros_like(self.F_state), trainable=False, name='momentum', dtype=tf.complex64)
        
        self.gather_indices, self.gather_friedel_mask = self._precompute_gather_indices()
        
        # Sparse mode active if TV is OFF (TV requires dense grid neighbors)
        self.sparse_mode = (self.tv_weight <= 0.0)
        
        if self.sparse_mode:
            print(f"Careleast: Sparse Mode Active.")
            if self.stochastic_points > 0 and (self.use_positivity or self.sparsity_weight > 0):
                print(f"Careleast: Using Stochastic Constraints (N={self.stochastic_points}).")
            else:
                print("Careleast: No spatial constraints.")

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

    def _gather_sigma_sparse(self):
        sigma_flat = tf.reshape(self.wilson_sigma_grid, (-1,))
        flat_indices = (self.gather_indices[:, 0] * self.ny * self.nz_half) + \
                       (self.gather_indices[:, 1] * self.nz_half) + \
                       self.gather_indices[:, 2]
        return tf.gather(sigma_flat, flat_indices)

    def _reconstruct_hkl_vectors(self, flat_indices):
        """Recover (h,k,l) from grid indices for DFT phase calculation."""
        iz = flat_indices % self.nz_half
        iy = (flat_indices // self.nz_half) % self.ny
        ix = flat_indices // (self.nz_half * self.ny)
        
        # Handle negative frequencies for h, k
        h = tf.where(ix < self.nx // 2, ix, ix - self.nx)
        k = tf.where(iy < self.ny // 2, iy, iy - self.ny)
        l = iz # RFFT l >= 0
        
        return tf.cast(tf.stack([h, k, l], axis=1), tf.float32)

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
        refl_id = tf.squeeze(self.get_refl_id(inputs), axis=-1)
        F_raw, _ = self._gather_F_sparse(self.F_state)
        F_obs_complex = tf.gather(self._apply_friedel(F_raw), refl_id, axis=1)
        F_abs_sq = tf.square(tf.abs(F_obs_complex))
        
        scale_dist = self.scaling_model(inputs)
        z_scale = scale_dist.sample(self.n_particles)
        if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
        z_scale = tf.where(tf.math.is_finite(z_scale), z_scale, tf.ones_like(z_scale))
        return z_scale * F_abs_sq

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]

        with tf.GradientTape() as tape:
            if not self.sparse_mode:
                tape.watch(self.F_state)
            
            # --- 1. Gather & Likelihood ---
            F_raw, flat_indices = self._gather_F_sparse(self.F_state)
            tape.watch(F_raw) 
            F_sym = self._apply_friedel(F_raw)
            
            refl_id = tf.squeeze(self.get_refl_id(data), axis=-1)
            F_obs_complex = tf.gather(F_sym, refl_id, axis=1)
            F_abs_sq = tf.square(tf.abs(F_obs_complex))
            
            scale_dist = self.scaling_model(data)
            z_scale = scale_dist.sample(self.n_particles)
            if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
            z_scale = tf.where(tf.math.is_finite(z_scale), z_scale, tf.ones_like(z_scale))
            
            ipred = z_scale * F_abs_sq
            ipred = tf.clip_by_value(ipred, 1e-12, 1e15)
            likelihood = self.likelihood(data)
            log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
            
            # --- 2. Prior & Constraints ---
            prior_energy = 0.0
            
            if self.sparse_mode:
                sigma_sparse = self._gather_sigma_sparse()
                F_sq_sparse = tf.square(tf.abs(F_raw))
                wilson_energy = 0.5 * tf.reduce_mean(F_sq_sparse, axis=1)
                prior_energy += self.prior_weight * wilson_energy
                
                # --- STOCHASTIC CONSTRAINTS (DFT) ---
                if self.stochastic_points > 0 and (self.use_positivity or self.sparsity_weight > 0):
                    # 1. Sample random fractional coordinates [0,1]
                    # Shape: (N_pts, 3)
                    r_frac = tf.random.uniform((self.stochastic_points, 3), dtype=tf.float32)
                    
                    # 2. Reconstruct HKLs
                    # NOTE: We need HKLs for F_raw (which are unique half-grid).
                    # But density is sum over ALL hkls (including friedel mates).
                    # Approximate: rho(r) = 2 * Real( sum_{h_unique} F_h * exp(-2pi i h.r) )
                    hkls = self._reconstruct_hkl_vectors(flat_indices) # (N_obs, 3)
                    
                    # 3. Compute Phase Shifts: theta = -2*pi * (r . h)
                    # (N_pts, 3) @ (3, N_obs) -> (N_pts, N_obs)
                    theta = -2.0 * np.pi * tf.matmul(r_frac, hkls, transpose_b=True)
                    exp_theta = tf.exp(tf.complex(0.0, theta))
                    
                    # 4. Compute Density: Sum_h (F_h * exp_theta)
                    # F_raw: (N_part, N_obs). Exp: (N_pts, N_obs)
                    # Result: (N_part, N_pts)
                    # Using Conjugate Transpose for dot product over N_obs
                    rho_stochastic = 2.0 * tf.math.real(tf.matmul(F_raw, exp_theta, transpose_b=True))
                    
                    # 5. Penalties
                    if self.use_positivity:
                        pos_loss = 10.0 * tf.reduce_mean(tf.square(tf.nn.relu(-rho_stochastic)), axis=1)
                        prior_energy += pos_loss
                        
                    if self.sparsity_weight > 0:
                        sparsity_loss = self.sparsity_weight * tf.reduce_mean(tf.abs(rho_stochastic), axis=1)
                        prior_energy += sparsity_loss

            else:
                # Dense FFT Mode
                F_sq = tf.square(tf.abs(self.F_state))
                wilson_energy = 0.5 * tf.reduce_mean(F_sq / tf.reshape(self.wilson_sigma_grid, (1, -1)), axis=1)
                prior_energy += self.prior_weight * wilson_energy
                
                if self.use_positivity or (self.tv_weight > 0.0):
                    F_full = self._expand_to_full(self.F_state)
                    rho = tf.math.real(tf.signal.ifft3d(F_full))
                    
                    if self.use_positivity:
                        prior_energy += 10.0 * tf.reduce_mean(tf.square(tf.nn.relu(-rho)), axis=[1,2,3])
                    if self.tv_weight > 0.0:
                        eps = 1e-6
                        dx = tf.sqrt(tf.square(rho - tf.roll(rho, 1, 1)) + eps)
                        dy = tf.sqrt(tf.square(rho - tf.roll(rho, 1, 2)) + eps)
                        dz = tf.sqrt(tf.square(rho - tf.roll(rho, 1, 3)) + eps)
                        prior_energy += self.tv_weight * tf.reduce_mean(dx + dy + dz, axis=[1,2,3])

            loss = -tf.reduce_mean(log_lik) + tf.reduce_mean(prior_energy)
            if tf.math.is_nan(loss): tf.print("\n[NaN DETECTED] Loss:", loss)

        # --- GRADIENTS ---
        if self.sparse_mode:
            grads = tape.gradient(loss, [F_raw] + self.scaling_model.trainable_variables)
            grads_F_raw, grads_scale = grads[0], grads[1:]

            # --- MODIFIED: FREEZE STRUCTURE ---
#            print("DEBUG: Structure Frozen. Refining Scale Only.")
#            grads = tape.gradient(loss, self.scaling_model.trainable_variables)
#            grads_F_raw = tf.zeros_like(F_raw) # <--- Force Zero Gradient
#            grads_scale = grads
            # ----------------------------------
        else:
            grads = tape.gradient(loss, [self.F_state] + self.scaling_model.trainable_variables)
            grads_F_raw, grads_scale = grads[0], grads[1:]
        
        safe_grads_scaling = []
        if isinstance(grads_scale, list):
             for g in grads_scale:
                safe_grads_scaling.append(self._nan_safe_real(g) if g is not None else None)
             self.optimizer.apply_gradients(zip(safe_grads_scaling, self.scaling_model.trainable_variables))
        
        # --- SGLD UPDATE ---
        raw_grad_norm = 0.0
        
        if self.sparse_mode:
            grads_F_raw = self._nan_safe_complex(grads_F_raw)
            sigma_sparse = self._gather_sigma_sparse()
            
            grid_indices = flat_indices
            p_indices = tf.range(self.n_particles, dtype=tf.int32)
            P_grid, G_grid = tf.meshgrid(p_indices, grid_indices, indexing='ij')
            scatter_indices = tf.stack([tf.reshape(P_grid, (-1,)), tf.reshape(G_grid, (-1,))], axis=1)
            
            m_curr_2d = tf.gather(self.momentum, flat_indices, axis=1)
            
            m_flat = tf.reshape(m_curr_2d, (-1,))
            g_flat = tf.reshape(grads_F_raw, (-1,))
            
            g_real = tf.math.real(g_flat)
            g_imag = tf.math.imag(g_flat)
            clipped_g, _ = tf.clip_by_global_norm([g_real, g_imag], 10.0)
            g_curr = tf.complex(clipped_g[0], clipped_g[1])
            
            N_obs = tf.shape(sigma_sparse)[0]
            sigma_tiled = tf.tile(sigma_sparse, [self.n_particles])
            temps_tiled = tf.repeat(self.temperatures, repeats=N_obs)
            
            grad_step = g_curr * tf.cast(sigma_tiled, tf.complex64)
            
            gs_real = tf.math.real(grad_step)
            gs_imag = tf.math.imag(grad_step)
            clipped_gs, _ = tf.clip_by_global_norm([gs_real, gs_imag], 100.0)
            grad_step = tf.complex(clipped_gs[0], clipped_gs[1])
            
            noise_scale = tf.sqrt(2.0 * self.learning_rate * (1.0 - self.friction) * temps_tiled)
            noise_std = noise_scale * tf.sqrt(sigma_tiled)
            
            nr = tf.random.normal(tf.shape(gs_real))
            ni = tf.random.normal(tf.shape(gs_imag))
            noise = tf.complex(nr, ni) * tf.cast(noise_std, tf.complex64)
            
            new_mom = (m_flat * self.friction) - (self.learning_rate * grad_step) + noise
            new_mom = self._nan_safe_complex(new_mom)
            
            self.momentum.scatter_nd_update(scatter_indices, new_mom)
            self.F_state.scatter_nd_add(scatter_indices, new_mom)
            
            raw_grad_norm = tf.reduce_sum(tf.abs(g_curr))
            
        else:
            grads_F_raw = self._nan_safe_complex(grads_F_raw)
            grad_F_real = tf.math.real(grads_F_raw)
            grad_F_imag = tf.math.imag(grads_F_raw)
            raw_grad_norm = tf.linalg.global_norm([grad_F_real, grad_F_imag])
            
            clipped_raw, _ = tf.clip_by_global_norm([grad_F_real, grad_F_imag], 10.0)
            grad_F = tf.complex(clipped_raw[0], clipped_raw[1])
            
            M = tf.reshape(self.wilson_sigma_grid, (1, -1))
            grad_step_complex = grad_F * tf.cast(M, tf.complex64)
            
            gs_real = tf.math.real(grad_step_complex)
            gs_imag = tf.math.imag(grad_step_complex)
            clipped_parts, _ = tf.clip_by_global_norm([gs_real, gs_imag], 100.0)
            grad_step = tf.complex(clipped_parts[0], clipped_parts[1])
            
            noise_scale = tf.sqrt(2.0 * self.learning_rate * (1.0 - self.friction) * self.temperatures)
            noise_std = noise_scale[:, None] * tf.sqrt(M)
            
            nr = tf.random.normal(tf.shape(gs_real))
            ni = tf.random.normal(tf.shape(gs_imag))
            noise = tf.complex(nr, ni) * tf.cast(noise_std, tf.complex64)
            
            new_momentum = (self.momentum * self.friction) - (self.learning_rate * grad_step) + noise
            new_momentum = self._nan_safe_complex(new_momentum)
            
            self.F_state.assign_add(new_momentum)
            self.momentum.assign(new_momentum)

        return {"loss": loss, "lik": tf.reduce_mean(log_lik), "NLL": -tf.reduce_mean(log_lik), "Grad Norm": raw_grad_norm}
    
    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        ipred = self(data)
        likelihood = self.likelihood(data)
        log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
        return {"loss": -tf.reduce_mean(log_lik), "NLL": -tf.reduce_mean(log_lik)}


class CareleastRealSpace(CareleastBase):
    # (Unchanged)
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
