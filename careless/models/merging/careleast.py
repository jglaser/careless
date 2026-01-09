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
            
            if use_custom_train_step and "Grad Norm" in _history:
                if not tf.math.is_finite(_history['Grad Norm']):
                    # Pass through, let user see the error or handle it
                    pass 
        return history


class CareleastReciprocalSpace(CareleastBase):
    """
    Reciprocal-Space SGLD.
    Optimizes Structure Factors F directly. 
    """
    def __init__(self, asu_collection, likelihood, scaling_model, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=False, prior_weight=0.1):
        super().__init__(asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight)
        
        self.use_positivity = use_positivity
        self.n_hkl = len(asu_collection.reciprocal_asus[0].lookup_table)
        self.wilson_sigma = self._precompute_wilson_sigma(b_factor)
        init_std = tf.sqrt(self.wilson_sigma / 2.0) 
        init_F = tf.random.normal((n_particles, self.n_hkl, 2), stddev=0.1 * init_std[None, :, None])
        self.F_state = tf.Variable(init_F, name='F_state', dtype=tf.float32)
        self.momentum = tf.Variable(tf.zeros_like(self.F_state), trainable=False, name='momentum')

    def _precompute_wilson_sigma(self, b_factor):
        dHKL = self.asu_collection.reciprocal_asus[0].dHKL
        sigma = np.exp(-b_factor * dHKL / 2.0).astype(np.float32)
        sigma /= sigma.max()
        return tf.constant(np.maximum(sigma, 1e-12))

    def _get_complex_F(self):
        return tf.complex(self.F_state[..., 0], self.F_state[..., 1])

    def call(self, inputs):
        refl_id = tf.squeeze(self.get_refl_id(inputs), axis=-1)
        F_batch_complex = tf.gather(self._get_complex_F(), refl_id, axis=1) 
        F_abs_sq = tf.square(tf.abs(F_batch_complex))
        
        scale_dist = self.scaling_model(inputs)
        z_scale = scale_dist.sample(self.n_particles)
        if len(z_scale.shape) == 1: z_scale = tf.expand_dims(z_scale, 0)
            
        return z_scale * F_abs_sq

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]

        with tf.GradientTape() as tape:
            # Watch Structure Factors
            tape.watch(self.F_state)
            
            ipred = self(data)
            ipred = tf.clip_by_value(ipred, 1e-12, 1e15) # Avoid log(0)
            
            likelihood = self.likelihood(data)
            log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
            
            # Prior: Wilson
            F_all = self._get_complex_F()
            F_sq_all = tf.square(tf.abs(F_all))
            wilson_energy = 0.5 * tf.reduce_sum(F_sq_all / self.wilson_sigma[None, :], axis=1)
            
            # Auxiliary Losses (Scale KL, etc)
            aux_loss = tf.reduce_sum(self.losses) if self.losses else 0.0

            # Total Potential
            potential = -log_lik + (self.prior_weight * wilson_energy) + aux_loss
            loss = tf.reduce_mean(potential)

        # Compute Gradients for EVERYTHING
        trainable_vars = [self.F_state] + self.scaling_model.trainable_variables
        grads = tape.gradient(potential, trainable_vars)
        
        grad_F = grads[0]
        grads_scale = grads[1:]
        
        # --- 1. Update Structure Factors (Manual SGLD) ---
        if isinstance(grad_F, tf.IndexedSlices):
             grad_F = tf.convert_to_tensor(grad_F)
        
        # NaN Guard
        grad_F = tf.where(tf.math.is_finite(grad_F), grad_F, tf.zeros_like(grad_F))
        raw_grad_norm = tf.linalg.global_norm([grad_F])
        
        # Precondition
        M = self.wilson_sigma[None, :, None]
        grad_step = grad_F * M
        
        # Clip AFTER preconditioning (Effective force clipping)
        grad_step = tf.clip_by_global_norm([grad_step], 100.0)[0][0]
        
        noise_scale = tf.sqrt(2.0 * self.learning_rate * (1.0 - self.friction) * self.temperatures)
        noise_std = noise_scale[:, None, None] * tf.sqrt(M)
        noise = tf.random.normal(tf.shape(self.F_state)) * noise_std
        
        new_momentum = (self.momentum * self.friction) - (self.learning_rate * grad_step) + noise
        new_momentum = tf.where(tf.math.is_finite(new_momentum), new_momentum, tf.zeros_like(new_momentum))
        
        self.momentum.assign(new_momentum)
        self.F_state.assign_add(new_momentum)
        
        # --- 2. Update Scales (Standard Optimizer) ---
        if self.optimizer and grads_scale:
            self.optimizer.apply_gradients(zip(grads_scale, self.scaling_model.trainable_variables))
        
        return {"loss": loss, "lik": tf.reduce_mean(log_lik), "NLL": loss, "Grad Norm": raw_grad_norm}
    
    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        ipred = self(data)
        likelihood = self.likelihood(data)
        log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
        return {"loss": -tf.reduce_mean(log_lik), "NLL": -tf.reduce_mean(log_lik)}


class CareleastRealSpace(CareleastBase):
    """
    Real-Space Stochastic Phasing Engine with Joint Scale Optimization.
    """
    def __init__(self, asu_collection, likelihood, scaling_model, grid_size, unit_cell, n_particles=4, b_factor=20.0, learning_rate=1e-3, friction=0.9, temperatures=None, use_positivity=True, prior_weight=0.1):
        super().__init__(asu_collection, likelihood, scaling_model, n_particles, learning_rate, friction, temperatures, prior_weight)
        self.grid_size = grid_size
        self.use_positivity = use_positivity
        
        self.n_pixels = tf.cast(tf.reduce_prod(grid_size), tf.float32)
        self.fft_scale = 1.0 / tf.sqrt(self.n_pixels)
        
        self.hkl_to_grid_indices = self._precompute_grid_indices()
        self.wilson_spectrum = self._precompute_wilson_spectrum(unit_cell, b_factor)

        # --- COLORED NOISE INITIALIZATION ---
        # 1. White noise in Reciprocal Space
        white_noise_F_real = tf.random.normal((n_particles, *grid_size))
        white_noise_F_imag = tf.random.normal((n_particles, *grid_size))
        white_noise_F = tf.complex(white_noise_F_real, white_noise_F_imag)
        
        # 2. Color by sqrt(Spectrum) to match physical B-factor
        # Note: We divide by fft_scale because _ifft_backward divides by it again,
        # but we want the variance to be 1/N. 
        # Actually, simpler: F ~ N(0, Spectrum). 
        # F_init = White * sqrt(Spectrum).
        # We need self.density such that FFT(density) has that variance.
        F_init = white_noise_F * tf.cast(tf.sqrt(self.wilson_spectrum), tf.complex64)
        
        # 3. Transform to Real Space
        rho_init = self._ifft_backward(F_init)
        
        self.density = tf.Variable(rho_init, name='density', dtype=tf.float32)
        self.momentum = tf.Variable(tf.zeros_like(self.density), trainable=False, name='momentum')

    def _precompute_grid_indices(self):
        lookup = self.asu_collection.reciprocal_asus[0].lookup_table
        hkls = lookup.get_hkls().astype(np.int32)
        nx, ny, nz = self.grid_size
        h_idx = tf.math.floormod(hkls[:, 0], nx)
        k_idx = tf.math.floormod(hkls[:, 1], ny)
        l_idx = tf.math.floormod(hkls[:, 2], nz)
        return tf.stack([h_idx, k_idx, l_idx], axis=1)

    def _precompute_wilson_spectrum(self, unit_cell, b_factor):
        nx, ny, nz = self.grid_size
        uc = unit_cell.parameters 
        KX, KY, KZ = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny), np.fft.fftfreq(nz), indexing='ij')
        s_sq = (KX * nx / uc[0])**2 + (KY * ny / uc[1])**2 + (KZ * nz / uc[2])**2
        spectrum = np.exp(-b_factor * s_sq / 2.0).astype(np.float32)
        spectrum /= spectrum.max()
        return tf.constant(np.maximum(spectrum, 1e-12), dtype=tf.complex64)

    def _fft_forward(self, density):
        d_complex = tf.complex(density, tf.zeros_like(density))
        return tf.signal.fft3d(d_complex) * tf.complex(self.fft_scale, 0.0)

    def _ifft_backward(self, F_grid):
        return tf.math.real(tf.signal.ifft3d(F_grid)) / self.fft_scale

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

        return z_scale * tf.square(tf.abs(F_obs))

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        with tf.GradientTape() as tape:
            tape.watch(self.density)
            
            ipred = self(data)
            ipred = tf.clip_by_value(ipred, 1e-12, 1e15)

            likelihood = self.likelihood(data)
            log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
            
            F_grid = self._fft_forward(self.density)
            F_sq = tf.square(tf.abs(F_grid))
            wilson_energy = 0.5 * tf.reduce_sum(F_sq / tf.abs(self.wilson_spectrum), axis=[1,2,3])
            
            prior_energy = self.prior_weight * wilson_energy
            
            if self.use_positivity:
                prior_energy += 1e3 * tf.reduce_sum(tf.square(tf.nn.relu(-self.density)), axis=[1,2,3])
            
            aux_loss = tf.reduce_sum(self.losses) if self.losses else 0.0
            potential = -log_lik + prior_energy + aux_loss
            loss = tf.reduce_mean(potential)

        trainable_vars = [self.density] + self.scaling_model.trainable_variables
        grads = tape.gradient(potential, trainable_vars)
        grad_rho = grads[0]
        grads_scale = grads[1:]

        # 1. Update Density (SGLD)
        grad_rho = tf.where(tf.math.is_finite(grad_rho), grad_rho, tf.zeros_like(grad_rho))
        
        # Precondition FIRST (Convert stiff forces to density updates)
        grads_k = self._fft_forward(grad_rho) * self.wilson_spectrum
        grads_smooth = self._ifft_backward(grads_k)
        
        # Clip AFTER preconditioning
        grads_smooth = tf.clip_by_global_norm([grads_smooth], 100.0)[0][0]
        
        noise_std = tf.sqrt(2.0 * self.learning_rate * (1.0 - self.friction) * self.temperatures)
        white_noise = tf.random.normal(shape=tf.shape(self.density))
        noise_colored = self._ifft_backward(self._fft_forward(white_noise) * tf.sqrt(self.wilson_spectrum))
        noise_term = noise_colored * noise_std[:, None, None, None]
        
        new_momentum = (self.momentum * self.friction) - (self.learning_rate * grads_smooth) + noise_term
        new_momentum = tf.where(tf.math.is_finite(new_momentum), new_momentum, tf.zeros_like(new_momentum))
        self.momentum.assign(new_momentum)
        self.density.assign_add(new_momentum)
        
        # 2. Update Scales (Standard Optimizer)
        if self.optimizer and grads_scale:
            self.optimizer.apply_gradients(zip(grads_scale, self.scaling_model.trainable_variables))
        
        # Log raw norm for debugging
        raw_grad_norm = tf.linalg.global_norm([grad_rho])
        
        return {"loss": loss, "lik": tf.reduce_mean(log_lik), "NLL": loss, "Grad Norm": raw_grad_norm}
    
    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        ipred = self(data)
        likelihood = self.likelihood(data)
        log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
        return {"loss": -tf.reduce_mean(log_lik), "NLL": -tf.reduce_mean(log_lik)}
