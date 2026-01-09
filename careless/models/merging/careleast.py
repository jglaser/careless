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
    Spectral SGLD.
    Optimizes Structure Factors F on a FULL grid using Complex-to-Complex FFTs.
    Includes Debugging and Robust Normalization.
    """
    def __init__(self, asu_collection, likelihood, scaling_model, grid_size, unit_cell, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=True, tv_weight=0.0, prior_weight=0.1, enforce_symmetry=True):
        super().__init__(asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight)
        
        self.grid_size = grid_size
        self.nx, self.ny, self.nz = grid_size
        self.enforce_symmetry = enforce_symmetry
        
        self.use_positivity = use_positivity
        self.tv_weight = tv_weight
        
        # Wilson Sigma on FULL Grid
        self.wilson_sigma_grid = self._precompute_wilson_sigma_grid(unit_cell, b_factor)
        
        # Initialize State (Complex F)
        # Reduced init_std to 0.001 to prevent initial explosion
        init_std = 0.001 
        init_F_real = tf.random.normal((n_particles, self.nx, self.ny, self.nz)) * init_std
        init_F_imag = tf.random.normal((n_particles, self.nx, self.ny, self.nz)) * init_std
        
        self.F_state = tf.Variable(tf.complex(init_F_real, init_F_imag), name='F_state', dtype=tf.complex64)
        self.momentum = tf.Variable(tf.zeros_like(self.F_state), trainable=False, name='momentum', dtype=tf.complex64)
        
        self.gather_indices = self._precompute_gather_indices()

    def _precompute_wilson_sigma_grid(self, unit_cell, b_factor):
        uc = unit_cell.parameters
        kx = np.fft.fftfreq(self.nx).astype(np.float32) * self.nx
        ky = np.fft.fftfreq(self.ny).astype(np.float32) * self.ny
        kz = np.fft.fftfreq(self.nz).astype(np.float32) * self.nz 
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        s_sq = (KX / uc[0])**2 + (KY / uc[1])**2 + (KZ / uc[2])**2
        sigma = np.exp(-b_factor * s_sq / 2.0).astype(np.float32)
        sigma /= sigma.max()
        # Increased clamp slightly to avoid 1e12 gradients
        return tf.constant(np.maximum(sigma, 1e-6))

    def _precompute_gather_indices(self):
        lookup = self.asu_collection.reciprocal_asus[0].lookup_table
        hkls = lookup.get_hkls().astype(np.int32)
        h = np.mod(hkls[:, 0], self.nx)
        k = np.mod(hkls[:, 1], self.ny)
        l = np.mod(hkls[:, 2], self.nz)
        indices = np.stack([h, k, l], axis=1)
        return tf.constant(indices, dtype=tf.int32)

    def _gather_F(self, F_grid):
        F_flat = tf.reshape(F_grid, (self.n_particles, -1))
        flat_indices = (self.gather_indices[:, 0] * self.ny * self.nz) + \
                       (self.gather_indices[:, 1] * self.nz) + \
                       self.gather_indices[:, 2]
        return tf.gather(F_flat, flat_indices, axis=1)

    def _symmetrize_F(self, F):
        F_rev = tf.reverse(F, axis=[1, 2, 3])
        F_rev = tf.roll(F_rev, shift=[1, 1, 1], axis=[1, 2, 3])
        return 0.5 * (F + tf.math.conj(F_rev))

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
        
        if self.enforce_symmetry:
            F_effective = self._symmetrize_F(self.F_state)
        else:
            F_effective = self.F_state
            
        F_all_refls = self._gather_F(F_effective)
        F_obs_complex = tf.gather(F_all_refls, refl_id, axis=1)
        F_abs_sq = tf.square(tf.abs(F_obs_complex))
        
        scale_dist = self.scaling_model(inputs)
        z_scale = scale_dist.sample(self.n_particles)
        if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
        
        z_scale = tf.where(tf.math.is_finite(z_scale), z_scale, tf.ones_like(z_scale))
        return z_scale * F_abs_sq

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]

        with tf.GradientTape() as tape:
            tape.watch(self.F_state)
            
            if self.enforce_symmetry:
                F_curr = self._symmetrize_F(self.F_state)
            else:
                F_curr = self.F_state

            # Likelihood
            refl_id = tf.squeeze(self.get_refl_id(data), axis=-1)
            F_all_refls = self._gather_F(F_curr)
            F_obs_complex = tf.gather(F_all_refls, refl_id, axis=1)
            F_abs_sq = tf.square(tf.abs(F_obs_complex))
            
            scale_dist = self.scaling_model(data)
            z_scale = scale_dist.sample(self.n_particles)
            if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
            
            # Guard against bad scales
            z_scale = tf.where(tf.math.is_finite(z_scale), z_scale, tf.ones_like(z_scale))
            
            ipred = z_scale * F_abs_sq
            ipred = tf.clip_by_value(ipred, 1e-12, 1e15)
            
            likelihood = self.likelihood(data)
            log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
            
            # Wilson Prior (Reduce MEAN to prevent overflow)
            F_sq = tf.square(tf.abs(F_curr))
            wilson_energy = 0.5 * tf.reduce_mean(F_sq / self.wilson_sigma_grid[None, ...], axis=[1,2,3])
            prior_energy = self.prior_weight * wilson_energy
            
            # Diagnostics placeholders
            pos_energy = 0.0
            tv_energy = 0.0
            
            # Real Space Constraints
            if self.use_positivity or (self.tv_weight > 0.0):
                rho_complex = tf.signal.ifft3d(F_curr)
                rho = tf.math.real(rho_complex) 
                
                if self.use_positivity:
                    # Reduced weight and MEAN
                    pos_energy = 10.0 * tf.reduce_mean(tf.square(tf.nn.relu(-rho)), axis=[1,2,3])
                    prior_energy += pos_energy
                    
                if self.tv_weight > 0.0:
                    eps = 1e-6
                    dx = tf.sqrt(tf.square(rho - tf.roll(rho, shift=1, axis=1)) + eps)
                    dy = tf.sqrt(tf.square(rho - tf.roll(rho, shift=1, axis=2)) + eps)
                    dz = tf.sqrt(tf.square(rho - tf.roll(rho, shift=1, axis=3)) + eps)
                    tv_energy = self.tv_weight * tf.reduce_mean(dx + dy + dz, axis=[1,2,3])
                    prior_energy += tv_energy

            loss = -tf.reduce_mean(log_lik) + tf.reduce_mean(prior_energy)
            
            # --- DEBUGGING NANs ---
            if tf.math.is_nan(loss):
                tf.print("\n[NaN DETECTED] Debugging:")
                tf.print("Loss:", loss)
                tf.print("LogLik (mean):", tf.reduce_mean(log_lik))
                tf.print("Wilson Energy:", tf.reduce_mean(wilson_energy))
                tf.print("Pos Energy:", tf.reduce_mean(pos_energy))
                tf.print("TV Energy:", tf.reduce_mean(tv_energy))
                tf.print("F_state max:", tf.reduce_max(tf.abs(self.F_state)))
                tf.print("z_scale max:", tf.reduce_max(z_scale))

        grads = tape.gradient(loss, [self.F_state] + self.scaling_model.trainable_variables)
        grad_F = grads[0]
        
        # Scaling Gradients Safe
        safe_grads_scaling = []
        if len(grads) > 1:
            for g in grads[1:]:
                if g is not None:
                    safe_grads_scaling.append(self._nan_safe_real(g))
                else:
                    safe_grads_scaling.append(None)
        
        # SGLD Update
        grad_F = self._nan_safe_complex(grad_F)
        grad_F_real = tf.math.real(grad_F)
        grad_F_imag = tf.math.imag(grad_F)
        raw_grad_norm = tf.linalg.global_norm([grad_F_real, grad_F_imag])
        
        clipped_raw, _ = tf.clip_by_global_norm([grad_F_real, grad_F_imag], 10.0)
        grad_F = tf.complex(clipped_raw[0], clipped_raw[1])
        
        M = self.wilson_sigma_grid[None, ...]
        grad_step_complex = grad_F * tf.cast(M, tf.complex64)
        
        gs_real = tf.math.real(grad_step_complex)
        gs_imag = tf.math.imag(grad_step_complex)
        clipped_parts, _ = tf.clip_by_global_norm([gs_real, gs_imag], 100.0)
        grad_step = tf.complex(clipped_parts[0], clipped_parts[1])
        
        noise_scale = tf.sqrt(2.0 * self.learning_rate * (1.0 - self.friction) * self.temperatures)
        noise_std = noise_scale[:, None, None, None] * tf.sqrt(M)
        
        noise_real = tf.random.normal(tf.shape(gs_real))
        noise_imag = tf.random.normal(tf.shape(gs_imag))
        noise = tf.complex(noise_real, noise_imag) * tf.cast(noise_std, tf.complex64)
        
        new_momentum = (self.momentum * self.friction) - (self.learning_rate * grad_step) + noise
        new_momentum = self._nan_safe_complex(new_momentum)
        
        self.F_state.assign_add(new_momentum)
        self.momentum.assign(new_momentum)

        if safe_grads_scaling:
            self.optimizer.apply_gradients(zip(safe_grads_scaling, self.scaling_model.trainable_variables))
        
        return {"loss": loss, "lik": tf.reduce_mean(log_lik), "NLL": -tf.reduce_mean(log_lik), "Grad Norm": raw_grad_norm}
    
    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        ipred = self(data)
        likelihood = self.likelihood(data)
        log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
        return {"loss": -tf.reduce_mean(log_lik), "NLL": -tf.reduce_mean(log_lik)}


class CareleastRealSpace(CareleastBase):
    """
    Real-Space Stochastic Phasing Engine.
    Uses Standard FFT. Reduced initialization & Normalized Energies.
    """
    def __init__(self, asu_collection, likelihood, scaling_model, grid_size, unit_cell, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=True, tv_weight=0.0, prior_weight=0.1):
        super().__init__(asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight)
        self.grid_size = grid_size
        self.use_positivity = use_positivity
        self.tv_weight = tv_weight
        
        self.n_pixels = tf.cast(tf.reduce_prod(grid_size), tf.float32)
        
        self.hkl_to_grid_indices = self._precompute_grid_indices()
        self.wilson_spectrum = self._precompute_wilson_spectrum(unit_cell, b_factor)

        # Reduced Init
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
            
            # Predict
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
            
            # Prior (Normalized MEAN)
            F_sq = tf.square(tf.abs(F_grid))
            wilson_energy = 0.5 * tf.reduce_mean(F_sq / tf.abs(self.wilson_spectrum), axis=[1,2,3])
            prior_energy = self.prior_weight * wilson_energy
            
            pos_energy = 0.0
            tv_energy = 0.0

            if self.use_positivity:
                pos_energy = 10.0 * tf.reduce_mean(tf.square(tf.nn.relu(-self.density)), axis=[1,2,3])
                prior_energy += pos_energy
            
            if self.tv_weight > 0.0:
                 eps = 1e-6
                 dx = tf.sqrt(tf.square(self.density - tf.roll(self.density, 1, axis=1)) + eps)
                 dy = tf.sqrt(tf.square(self.density - tf.roll(self.density, 1, axis=2)) + eps)
                 dz = tf.sqrt(tf.square(self.density - tf.roll(self.density, 1, axis=3)) + eps)
                 tv_energy = self.tv_weight * tf.reduce_mean(dx + dy + dz, axis=[1,2,3])
                 prior_energy += tv_energy

            loss = -tf.reduce_mean(log_lik) + tf.reduce_mean(prior_energy)
            
            if tf.math.is_nan(loss):
                tf.print("\n[NaN DETECTED] Debugging RealSpace:")
                tf.print("Loss:", loss)
                tf.print("LogLik:", tf.reduce_mean(log_lik))
                tf.print("Prior Energy:", tf.reduce_mean(prior_energy))
                tf.print("Density Max:", tf.reduce_max(tf.abs(self.density)))
                tf.print("Z Scale Max:", tf.reduce_max(z_scale))

        grads = tape.gradient(loss, [self.density] + self.scaling_model.trainable_variables)
        grad_rho = grads[0]
        
        safe_grads_scaling = []
        if len(grads) > 1:
            for g in grads[1:]:
                if g is not None:
                    safe_grads_scaling.append(self._nan_safe_real(g))
                else:
                    safe_grads_scaling.append(None)
        
        # SGLD
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
