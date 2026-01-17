import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from careless.models.priors.base import Prior
from careless.models.merging.surrogate_posteriors import RiceWoolfson
from careless.utils.density import build_symmetry_map_gemmi, probed_ifft
import math


class ReferencePrior(Prior):
    """
    A Prior class with a `log_prob` implementation that returns zeros for unobserved miller indices.
     - This class is not meant to be used directly. 
     - Extensions of this class must set the `base_dist` attribute with a tfd.Distribution or similar object with
       a `.log_prob(values)` method.
    """
    base_dist = None
    def __init__(self, observed=None):
        super().__init__()
        if observed is None:
            self.idx = None
        else:
            idx = tf.where(observed)
            self.idx = tf.reshape(idx, (-1,))

    def mean(self):
        """ This just passes through to self.base_dist. """
        return self.base_dist.mean()

    def stddev(self):
        """ This just passes through to self.base_dist. """
        return self.base_dist.stddev()

    def log_prob(self, values):
        if self.idx is None:
            return self.base_dist.log_prob(values)
        obs = tf.gather(values, self.idx, axis=-1)
        log_prob = self.base_dist.log_prob(obs)

        return tf.transpose(tf.scatter_nd(
            self.idx[...,None], 
            tf.transpose(log_prob), 
            values.shape[::-1]
        ))

class LaplaceReferencePrior(ReferencePrior):
    """
    A Laplacian prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, observed=None):
        super().__init__()
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)/math.sqrt(2.)
        self.base_dist = tfd.Laplace(loc, scale)

class NormalReferencePrior(ReferencePrior):
    """
    A Normal prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, observed=None):
        super().__init__()
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        self.base_dist = tfd.Normal(loc, scale)

class StudentTReferencePrior(ReferencePrior):
    """
    A Student's T prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, dof, observed=None):
        super().__init__()
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        dof : float
            degrees of freedom for the student's t distribution.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        self.base_dist = tfd.StudentT(dof, loc, scale)

class RiceWoolfsonReferencePrior(ReferencePrior):
    """
    A Rice Woolfson prior distribution centered at empirical structure factor amplitudes derived from a conventional experiment.
    """
    def __init__(self, Fobs, SigFobs, centric, observed=None):
        super().__init__()
        """
        Parameters
        ----------
        Fobs : array
            numpy array or tf.Tensor containing observed structure factors amplitudes from a reference structure.
        SigFobs : array
            numpy array or tf.Tensor containing error estimates for structure factors amplitudes from a reference structure.
        centric : array
            boolean numpy array or tf.Tensor which has True for all centric reflections.
        observed : array (optional)
            boolean numpy array or tf.Tensor which has True for all observed miller indices.
        """
        super().__init__(observed)
        loc = np.array(Fobs, dtype=np.float32)
        scale = np.array(SigFobs, dtype=np.float32)
        self.base_dist = RiceWoolfson(loc, scale, centric)

class SparseRealSpacePrior(Prior):
    """
    Applies real-space constraints (Positivity, Sparsity, Total Variation)
    using Stochastic Non-Uniform DFT.
    """
    def __init__(self, miller_indices, n_points=4096, positivity_weight=1.0, sparsity_weight=0.0, tv_weight=0.0):
        """
        miller_indices : array-like (N, 3)
            Miller indices corresponding to the unique structure factors.
        tv_weight : float
            Weight for Total Variation regularization (penalizes density gradients).
        """
        super().__init__()
        self.hkls = tf.cast(miller_indices, tf.float32)
        self.n_points = n_points
        self.positivity_weight = positivity_weight
        self.sparsity_weight = sparsity_weight
        self.tv_weight = tv_weight

        # Precompute gradient factors for TV: -2*pi*i * (h,k,l)
        # Shape: (N_refls, 3). We use complex64 for the multiplication.
        if self.tv_weight > 0:
            self.grad_factors = tf.cast(self.hkls, tf.complex64) * tf.complex(0., -2.0 * np.pi)

    def log_prob(self, z_complex):
        """
        z_complex: Tensor (Batch, N_refls) of complex structure factors.
        """
        # 1. Sample random fractional coordinates r in [0, 1]
        # Shape: (N_points, 3)
        r_frac = tf.random.uniform((self.n_points, 3), dtype=tf.float32)

        # 2. Compute Phase Shifts: theta = -2*pi * (r . h)
        # (N_points, 3) @ (3, N_refls) -> (N_points, N_refls)
        theta = -2.0 * np.pi * tf.matmul(r_frac, self.hkls, transpose_b=True)
        exp_theta = tf.exp(tf.complex(0.0, theta))

        # 3. Compute Density: rho(r)
        # (Batch, N_refls) @ (N_refls, N_points) -> (Batch, N_points)
        rho = 2.0 * tf.math.real(tf.matmul(z_complex, exp_theta, transpose_b=True))

        total_log_prob = 0.0

        # --- A. Positivity (ReLU) ---
        if self.positivity_weight > 0:
            neg_density = tf.nn.relu(-rho)
            pos_loss = tf.reduce_mean(tf.square(neg_density), axis=1)
            total_log_prob -= self.positivity_weight * pos_loss

        # --- B. Sparsity (L1) ---
        if self.sparsity_weight > 0:
            sparsity_loss = tf.reduce_mean(tf.abs(rho), axis=1)
            total_log_prob -= self.sparsity_weight * sparsity_loss

        # --- C. Total Variation (Gradient L1) ---
        if self.tv_weight > 0:
            # We need to compute Sum [ F_h * (-2pi i h) * exp(...) ]
            # Broadcast z_complex against the 3 gradient dimensions (h,k,l)
            # z_complex: (Batch, N_refls)
            # grad_factors: (N_refls, 3)
            # z_grad: (Batch, N_refls, 3)
            z_grad = tf.expand_dims(z_complex, -1) * tf.expand_dims(self.grad_factors, 0)

            # Perform DFT for the vector field (3 components)
            # Einsum:
            #   b=batch, r=reflections, d=xyz_dim, p=points
            #   input (b,r,d) and (p,r) [transpose of exp_theta] -> output (b,p,d)
            grad_field_complex = tf.einsum('brd,pr->bpd', z_grad, exp_theta)

            # Real space gradient vector field
            grad_field = 2.0 * tf.math.real(grad_field_complex)

            # Compute magnitude of gradient at each point: sqrt(dx^2 + dy^2 + dz^2)
            grad_norm = tf.norm(grad_field, axis=-1)

            # Average over points (Monte Carlo integral of |Grad(rho)|)
            tv_loss = tf.reduce_mean(grad_norm, axis=1)

            total_log_prob -= self.tv_weight * tv_loss

        return total_log_prob

class DenseRealSpacePrior(Prior):
    """
    Applies Real-Space constraints (TV, Positivity, Sparsity) by expanding
    ASU coefficients to a P1 grid and performing a dense 3D FFT.
    """
    def __init__(self, asu_collection, grid_size, tv_weight=0.0, positivity_weight=0.0, sparsity_weight=0.0):
        super().__init__()
        self.tv_weight = tv_weight
        self.positivity_weight = positivity_weight
        self.sparsity_weight = sparsity_weight
        self.grid_size = grid_size # (nz, ny, nx)
        self.nx, self.ny, self.nz = grid_size
        self.nz_half = self.nz // 2 + 1

        # 1. Build Symmetry Expansion Map (Reuse existing logic)
        # Note: We take the first ASU from the collection. Assumes merged dataset logic.
        sg_symbol = asu_collection.reciprocal_asus[0].spacegroup.hm
        cell = asu_collection.reciprocal_asus[0].cell

        gather_ids, phase_shifts, conj_flags, n_unique = build_symmetry_map_gemmi(
            cell, sg_symbol, (self.nx, self.ny, self.nz_half)
        )

        self.asu_to_grid_indices = tf.constant(gather_ids, dtype=tf.int32)
        self.asu_phase_shifts = tf.constant(phase_shifts, dtype=tf.complex64)
        self.asu_conj_flags = tf.constant(conj_flags, dtype=tf.bool)

    def _expand_to_full(self, F_flat_state, batch_size):
        # Reshape to Half-Grid
        F_half = tf.reshape(F_flat_state, (batch_size, self.nx, self.ny, self.nz_half))
        limit = -1 if (self.nz % 2 == 0) else None

        # Hermitian Symmetry Expansion for Real Output
        middle_slice = F_half[..., 1:limit]
        rev_slice = tf.reverse(middle_slice, axis=[1, 2, 3])
        rev_slice = tf.roll(rev_slice, shift=[1, 1, 0], axis=[1, 2, 3])
        redundant_part = tf.math.conj(rev_slice)

        return tf.concat([F_half, redundant_part], axis=-1)

    def log_prob(self, z_complex):
        """
        z_complex: (Batch, N_asu)
        """
        batch_size = tf.shape(z_complex)[0]

        # 1. Expand ASU -> Half Grid
        # Gather (Batch, Grid_Flat) from (Batch, ASU)
        F_gathered = tf.gather(z_complex, self.asu_to_grid_indices, axis=1)
        F_expanded = tf.where(self.asu_conj_flags, tf.math.conj(F_gathered), F_gathered)
        F_grid_flat = F_expanded * self.asu_phase_shifts

        # 2. Expand Half Grid -> Full Grid
        F_full = self._expand_to_full(F_grid_flat, batch_size)

        # 3. IFFT to Real Space Density
        # We use the custom gradient 'probed_ifft' for safety, or standard ifft3d
        rho_complex = probed_ifft(F_full)

        # Scaling (Matches TF's 1/N normalization in IFFT)
        total_grid_points = tf.cast(self.nx * self.ny * self.nz, tf.float32)
        rho = tf.math.real(rho_complex) * total_grid_points

        # 4. Calculate Constraints (Negative Log Prob = Loss)
        prior_log_prob = 0.0

        # A. Sparsity (L1)
        if self.sparsity_weight > 0:
            l1_loss = tf.reduce_mean(tf.abs(rho), axis=[1,2,3])
            prior_log_prob -= self.sparsity_weight * l1_loss

        # B. Positivity (ReLU)
        if self.positivity_weight > 0:
            pos_loss = tf.reduce_mean(tf.square(tf.nn.relu(-rho)), axis=[1,2,3])
            prior_log_prob -= self.positivity_weight * pos_loss

        # C. Total Variation (Anisotropic suppression)
        if self.tv_weight > 0:
            eps = 1e-6
            dx = rho - tf.roll(rho, shift=1, axis=1)
            dy = rho - tf.roll(rho, shift=1, axis=2)
            dz = rho - tf.roll(rho, shift=1, axis=3)
            grad_mag = tf.sqrt(tf.square(dx) + tf.square(dy) + tf.square(dz) + eps)
            tv_loss = tf.reduce_mean(grad_mag, axis=[1,2,3])
            prior_log_prob -= self.tv_weight * tv_loss

        return prior_log_prob
