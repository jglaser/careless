from careless.models.base import BaseModel
import tensorflow as tf

class Prior(BaseModel):
    """Base class for prior distributions on merged normalized structure factor amplitudes."""
    def log_prob(self):
        raise NotImplementedError("No log_prob method defined. All Priors must implement a log_prob method")

class JointPrior(Prior):
    """
    Combines a Wilson Prior (on amplitudes) with a Real-Space Prior (on phases/density).
    """
    def __init__(self, wilson_prior, sparse_prior):
        """
        wilson_prior : careless.models.priors.wilson.WilsonPrior
            Computes log_prob per reflection based on amplitude.
        sparse_prior : careless.models.priors.empirical.SparseRealSpacePrior
            Computes global log_prob based on Fourier Transform of complex factors.
        """
        super().__init__()
        self.wilson_prior = wilson_prior
        self.sparse_prior = sparse_prior

    def log_prob(self, z_complex):
        """
        z_complex : tf.Tensor (Batch, n_refls)
            Complex structure factors.
        """
        # 1. Wilson Prior Term (Amplitude only)
        # Wilson log_prob returns shape (Batch, n_refls, 1) or (Batch, n_refls)
        amp = tf.abs(z_complex)
        wilson_log_prob = self.wilson_prior.log_prob(amp)

        # Sum over reflections to get a global log_prob per batch member
        # Shape: (Batch,)
        wilson_term = tf.reduce_sum(wilson_log_prob, axis=-1)
        if len(wilson_term.shape) > 1:
            wilson_term = tf.reduce_sum(wilson_term, axis=-1)

        # 2. Sparse Prior Term (Full Complex / Real Space)
        # Returns shape (Batch,)
        sparse_term = self.sparse_prior.log_prob(z_complex)

        # 3. Combine
        return wilson_term + sparse_term
