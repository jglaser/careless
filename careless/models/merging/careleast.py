import tensorflow as tf
import tf_keras as tfk
import numpy as np
import reciprocalspaceship as rs
from careless.models.base import BaseModel
from tqdm.autonotebook import tqdm

class CareleastRealSpace(tfk.models.Model, BaseModel):
    """
    Real-Space Stochastic Phasing Engine with Parallel Tempering.
    """
    def __init__(self, 
                 asu_collection, 
                 likelihood, 
                 scaling_model, 
                 grid_size, 
                 unit_cell, 
                 n_particles=4, 
                 b_factor=20.0, 
                 learning_rate=1e-3, 
                 friction=0.9,
                 temperatures=None):
        super().__init__()
        self.asu_collection = asu_collection
        self.likelihood = likelihood
        self.scaling_model = scaling_model
        self.grid_size = grid_size
        self.n_particles = n_particles
        self.learning_rate = learning_rate
        self.friction = friction
        
        # FFT Normalization Factor (1/sqrt(N))
        self.n_pixels = tf.cast(tf.reduce_prod(grid_size), tf.float32)
        self.fft_scale = 1.0 / tf.sqrt(self.n_pixels)
        
        if temperatures is None:
            temperatures = np.geomspace(1.0, 2.5, n_particles)
        else:
            temperatures = np.array(temperatures, dtype=np.float32)
            
        assert len(temperatures) == n_particles
        self.temperatures = tf.Variable(temperatures, trainable=False, dtype=tf.float32, name="temperatures")

        # Initialize Density (Float32)
        self.density = tf.Variable(
            tf.random.normal((n_particles, *grid_size), stddev=1e-3), 
            name='density', 
            dtype=tf.float32
        )
        self.momentum = tf.Variable(tf.zeros_like(self.density), trainable=False, name='momentum')

        # Precompute kernels
        self.hkl_to_grid_indices = self._precompute_grid_indices()
        self.wilson_spectrum = self._precompute_wilson_spectrum(unit_cell, b_factor)

    def _precompute_grid_indices(self):
        lookup = self.asu_collection.reciprocal_asus[0].lookup_table
        hkls = lookup.get_hkls().astype(np.int32)
        
        nx, ny, nz = self.grid_size
        h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
        
        h_idx = tf.math.floormod(h, nx)
        k_idx = tf.math.floormod(k, ny)
        l_idx = tf.math.floormod(l, nz)
        
        return tf.stack([h_idx, k_idx, l_idx], axis=1)

    def _precompute_wilson_spectrum(self, unit_cell, b_factor):
        nx, ny, nz = self.grid_size
        uc = unit_cell.parameters 
        
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        kz = np.fft.fftfreq(nz)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        inv_a = 1.0 / uc[0]
        inv_b = 1.0 / uc[1]
        inv_c = 1.0 / uc[2]
        
        s_sq = (KX * inv_a * nx)**2 + (KY * inv_b * ny)**2 + (KZ * inv_c * nz)**2
        
        spectrum = np.exp(-b_factor * s_sq / 4.0).astype(np.float32)
        spectrum /= spectrum.max()
        spectrum = np.maximum(spectrum, 1e-12)
        
        return tf.constant(spectrum, dtype=tf.complex64)

    def _fft_forward(self, density):
        # FIX: Use tf.complex(r, 0) instead of tf.cast(r, complex) to avoid gradient warnings
        # density is float32, we create complex64
        density_complex = tf.complex(density, tf.zeros_like(density))
        f = tf.signal.fft3d(density_complex)
        
        # Scale is float, make it complex explicitly
        scale_complex = tf.complex(self.fft_scale, 0.0)
        return f * scale_complex

    def _ifft_backward(self, F_grid):
        # F_grid is complex
        rho_complex = tf.signal.ifft3d(F_grid)
        rho_real = tf.math.real(rho_complex)
        return rho_real / self.fft_scale

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
        
        if len(z_scale.shape) == 1: 
             z_scale = tf.expand_dims(z_scale, 0)

        # F_obs is complex. Abs returns float. Square returns float.
        F_abs_sq = tf.square(tf.abs(F_obs))
        I_pred = z_scale * F_abs_sq
        return I_pred

    def _compute_potential_and_grads(self, data):
        with tf.GradientTape() as tape:
            tape.watch(self.density)
            
            ipred = self(data)
            likelihood = self.likelihood(data)
            log_lik = tf.reduce_sum(likelihood.log_prob(ipred), axis=-1)
            
            neg_density = tf.nn.relu(-self.density)
            constraint_loss = 1e3 * tf.reduce_sum(tf.square(neg_density), axis=[1,2,3])
            
            potential = -log_lik + constraint_loss
            loss = tf.reduce_mean(potential)

        grads = tape.gradient(potential, self.density)
        grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
        return loss, potential, log_lik, grads

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]

        loss, potential, log_lik, grads = self._compute_potential_and_grads(data)
        
        # Preconditioning
        grads_k = self._fft_forward(grads)
        grads_k_smooth = grads_k * self.wilson_spectrum
        grads_smooth = self._ifft_backward(grads_k_smooth)
        
        # SGHMC Update
        noise_std = tf.sqrt(2.0 * self.learning_rate * (1.0 - self.friction) * self.temperatures)
        white_noise = tf.random.normal(shape=tf.shape(self.density))
        colored_noise_k = self._fft_forward(white_noise) * tf.sqrt(self.wilson_spectrum)
        colored_noise = self._ifft_backward(colored_noise_k)
        noise_term = colored_noise * noise_std[:, None, None, None]
        
        new_momentum = (self.momentum * self.friction) - (self.learning_rate * grads_smooth) + noise_term
        self.momentum.assign(new_momentum)
        self.density.assign_add(new_momentum)
        
        return {
            "loss": loss, 
            "lik": tf.reduce_mean(log_lik),
            "NLL": loss 
        }

    def test_step(self, data):
        if isinstance(data, tuple): data = data[0]
        loss, _, log_lik, _ = self._compute_potential_and_grads(data)
        return {"loss": loss, "lik": tf.reduce_mean(log_lik), "NLL": loss}

    def train_step_with_gradient_norm(self, data=None):
        if data is None: x = self.data 
        elif isinstance(data, tuple): x = data[0]
        else: x = data
        
        loss, potential, log_lik, grads = self._compute_potential_and_grads(x)
        grad_norm = tf.linalg.global_norm([grads])
        
        metrics = self.train_step((x,))
        metrics["Grad Norm"] = grad_norm
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
                    print("Encountered numerical issues (Inf/NaN gradients), terminating early!")
                    break
        return history
