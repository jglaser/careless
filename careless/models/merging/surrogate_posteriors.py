from careless.utils.distributions import Rice,FoldedNormal
from tensorflow_probability.python.internal.special_math import ndtr
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.bijectors import masked_autoregressive_default_template
import tensorflow as tf
import tf_keras as tfk
import numpy as np

class SurrogatePosterior(tfk.models.Model):
    """ The base class for learnable variational distributions over structure factor amplitudes. """
    def __init__(self, distribution=None, **kwargs):
        super().__init__(**kwargs)
        self.distribution = distribution

    def sample(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self.distribution.log_prob(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.distribution.mean(*args, **kwargs)

    def stddev(self, *args, **kwargs):
        return self.distribution.stddev(*args, **kwargs)

    def parameter_properties(self, *args, **kwargs):
        return self.distribution.parameter_properties(*args, **kwargs)

    def moment_4(self):
        raise NotImplementedError("The fourth moment of this distribution is not implemented yet.")

    @property
    def parameters(self):
        return self.distribution.parameters


#This is a temporary workaround for tfd.TruncatedNormal which has a bug in sampling
#2020-10-30: This should be removed if this issue is fixed: https://github.com/tensorflow/probability/issues/1149
#
#2020-11-01: On second thought, this may not be fixed unless they git rid of the current rejection sampler based 
# implementation. See https://github.com/tensorflow/probability/issues/518, for additional issues.
class TruncatedNormal(SurrogatePosterior):
    def __init__(self, loc, scale, low, high, validate_args=False, allow_nan_stats=True, name='TruncatedNormal', **kwargs):
        distribution = tfd.TruncatedNormal(loc, scale, low, high, validate_args, allow_nan_stats, name)
        super().__init__(distribution, **kwargs)

    def sample(self, *args, **kwargs):
        s = self.distribution.sample(*args, **kwargs)
        low = self.distribution.low
        return tf.maximum(low, s)

    def _tf_moment_4(self, high=None):
        from tensorflow_probability.python.internal.special_math import ndtr
        if high is None:
            high = self.distribution.high
        a,b = self.distribution.low,high
        mu,sigma = self.distribution.loc, self.distribution.scale
        z_b = (b-mu)/sigma
        z_a = (a-mu)/sigma
        
        norm = tfd.Normal(0., 1.) #Standard Normal
        if b==np.inf:
            bterm=0.
        else:
            bterm = (b*b*b + b*b*mu + b*mu*mu + sigma*sigma*(3*b + 5*mu) + mu*mu*mu)*norm.prob(z_b) 
        aterm = (a*a*a+a*a*mu+a*mu*mu+sigma*sigma*(3*a+5*mu)+mu*mu*mu)*norm.prob(z_a)

        num = bterm - aterm
        den = ndtr(z_b) - ndtr(z_a)
        return mu*mu*mu*mu + 6*mu*mu*sigma*sigma + 3*sigma*sigma*sigma*sigma- sigma*num/den

    def _scipy_moment_4(self, high):
        from scipy.stats import truncnorm
        loc,scale = self.distribution.loc, self.distribution.scale
        low = self.distribution.low.numpy()
        if high is None:
            high = self.distribution.high.numpy()
        a, b = (low - loc) / scale, (high - loc) / scale
        mom4 = truncnorm.moment(4, a, b, loc, scale)
        return mom4

    def moment_4(self, high=np.inf, method='scipy'):
        """
        Calculate the fourth moment of this distribution. This is based on the formula here: 
        https://people.smp.uq.edu.au/YoniNazarathy/teaching_projects/studentWork/EricOrjebin_TruncatedNormalMoments.pdf

        Parameters
        ----------
        high : float (optional)
            The high parameter to use for the distribution. By default use inf.
        method : str (optional)
            Either 'scipy' or 'tf'
        """
        if method=='scipy':
            return self._scipy_moment_4(high)
        elif method == 'tf':
            return self._tf_moment_4(high)
        else:
            raise ValueError(f"Unknown method {method} for computing moment_4")

    @classmethod
    def from_loc_and_scale(cls, loc, scale, low=0., high=1e10, scale_shift=1e-7):
        """
        Instantiate a learnable distribution with good default bijectors.

        loc : array
            The initial location of the distribution
        scale : array
            The initial scale parameter of the distribution
        low : float or array (optional)
            The lower limit of the support for the distribution.
        high : float or array (optional)
            The upper limit of the support for the distribution.
        scale_shift : float (optional)
            A small constant added to the scale to increase numerical stability.
        """
        loc   = tfp.util.TransformedVariable(
            loc,
            tfb.Exp(),
        )
        scale = tfp.util.TransformedVariable(
            scale,
            tfb.Chain([
                tfb.Shift(scale_shift),
                tfb.Exp(),
            ]),
        )
        return cls(loc, scale, low, high)

class RiceWoolfson(tfd.Distribution):
    def __init__(self, loc, scale, centric):
        """
        This is a hybrid distribution to parameterize posteriors over structure factors. 
        It uses the Rice distribution to model acentric structure factors and
        the Folded normal or "Woolfson" distribution to model centrics. 

        Parameters
        ----------
        loc : array (float)
            location parameter for the distributions
        scale : array (float)
            scale parameter for the distributions
        centric : array (float;bool)
            Array that same length as loc and scale that is 1./True for centric reflections and 0./False
        """
        self._loc   = tensor_util.convert_nonref_to_tensor(loc, dtype=tf.float32)
        self._scale = tensor_util.convert_nonref_to_tensor(scale, dtype=tf.float32)
        self._centric = np.array(centric, dtype=bool)
        self._woolfson = FoldedNormal(self._loc, self._scale)
        self._rice = Rice(self._loc, self._scale)
        self.eps = np.finfo(np.float32).eps

    def mean(self):
        return tf.where(self._centric, self._woolfson.mean(), self._rice.mean())

    def variance(self):
        return tf.where(self._centric, self._woolfson.variance(), self._rice.variance())

    def stddev(self):
        return tf.where(self._centric, self._woolfson.stddev(), self._rice.stddev())

    def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
        return tf.where(self._centric, self._woolfson.sample(sample_shape, seed, name, **kwargs)+self.eps, self._rice.sample(sample_shape, seed, name, **kwargs))

    def log_prob(self, x):
        return tf.where(self._centric, self._woolfson.log_prob(x), self._rice.log_prob(x))

    def prob(self, x):
        return tf.where(self._centric, self._woolfson.prob(x), self._rice.prob(x))

class FlowPosterior(SurrogatePosterior):
    """
    A Surrogate Posterior parameterized by an Inverse Autoregressive Flow (IAF).
    IAF allows for parallel sampling, which is critical for the speed of Variational Inference.
    """
    def __init__(self, loc, scale, depth=2, hidden_units=16, inference_samples=100, name='FlowPosterior', **kwargs):
        """
        Parameters
        ----------
        loc : array
            Initial location parameter for the base distribution.
        scale : array
            Initial scale parameter for the base distribution.
        depth : int
            Number of autoregressive layers.
        hidden_units : int
            Width of the neural network in each layer.
        """
        n_dims = loc.shape[0]

        # 1. Create variables LOCALLY first
        base_loc = tf.Variable(loc, name='base_loc')
        base_scale = tfp.util.TransformedVariable(scale, tfb.Softplus(), name='base_scale')
        base_dist = tfd.MultivariateNormalDiag(loc=base_loc, scale_diag=base_scale)

        bijectors = []
        for i in range(depth):
            # Standard MAF is slow for sampling (O(D)).
            # Inverted MAF (IAF) is fast for sampling (O(1)).
            maf = tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=masked_autoregressive_default_template(
                    hidden_layers=[hidden_units, hidden_units]
                )
            )
            bijectors.append(tfb.Invert(maf))

            # Permute to mix dimensions (Permute is fast both ways)
            bijectors.append(tfb.Permute(permutation=np.random.permutation(n_dims)))

        bijectors.append(tfb.Softplus())

        chain = tfb.Chain(bijectors)
        distribution = tfd.TransformedDistribution(distribution=base_dist, bijector=chain)

        # 3. Initialize superclass
        super().__init__(distribution, name=name, **kwargs)
        self.base_loc = base_loc
        self.base_scale = base_scale
        self.inference_samples = inference_samples

    @property
    def parameters(self):
        return {
            'loc': self.base_loc,
            'scale': self.base_scale
        }

    def parameter_properties(self, dtype=tf.float32, num_classes=None):
        return {
            'loc': tfp.util.ParameterProperties(),
            'scale': tfp.util.ParameterProperties()
        }

    @tf.function
    def mean(self, n_samples=None):
        if n_samples is None: n_samples = self.inference_samples
        return tf.reduce_mean(self.distribution.sample(n_samples), axis=0)

    @tf.function
    def stddev(self, n_samples=None):
        if n_samples is None:
            n_samples = self.inference_samples
        return tf.math.reduce_std(self.distribution.sample(n_samples), axis=0)

    @tf.function
    def moment_4(self, n_samples=None, **kwargs):
        if n_samples is None:
            n_samples = self.inference_samples
        samples = self.distribution.sample(n_samples)
        return tf.reduce_mean(tf.pow(samples, 4), axis=0)

    @classmethod
    def from_loc_and_scale(cls, loc, scale, depth=2, hidden_units=16, inference_samples=100, **kwargs):
        return cls(loc, scale, depth=depth, hidden_units=hidden_units, inference_samples=inference_samples, **kwargs)

class SplineNet(tf.keras.layers.Layer):
    """
    Helper Layer for Rational Quadratic Splines.
    Supports 'Global Mixing' (Triangular Map) via Low-Rank Factorization.
    """
    def __init__(self, hidden_units, bins, kind, spline_range, n_refls=None, mixing_rank=None,
                 min_bin_width=1e-2, min_bin_height=1e-2, min_derivative=1e-2, name=None):
        super().__init__(name=name)
        self.bins = bins
        self.kind = kind
        self.spline_range = spline_range
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        # Local (Independent) Convolutions
        self.conv1 = tf.keras.layers.Conv1D(hidden_units, kernel_size=1, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(hidden_units, kernel_size=1, activation='relu')

        # Output projection
        if kind == 'slopes':
            self.conv_out = tf.keras.layers.Conv1D(bins - 1, kernel_size=1, kernel_initializer='zeros')
            self.activation = tf.keras.layers.Activation('softplus')
        else:
            self.conv_out = tf.keras.layers.Conv1D(bins, kernel_size=1, kernel_initializer='zeros')

        # Global Mixing
        self.use_global = False
        if n_refls is not None and n_refls < 20000:
            self.use_global = True
            
            # --- FIX: Use Physical Rank if provided, else heuristic ---
            if mixing_rank is not None:
                rank = mixing_rank
            else:
                rank = int(np.sqrt(n_refls))
                rank = max(4, min(rank, 64))
            
            # Low Rank Matrices
            self.mix_U = tf.keras.layers.Dense(rank, kernel_initializer='glorot_uniform', use_bias=False, name='mix_U')
            self.mix_V = tf.keras.layers.Dense(n_refls, kernel_initializer='zeros', use_bias=False, name='mix_V')
            self.ln = tf.keras.layers.LayerNormalization(axis=1)

    def call(self, inputs):
        # inputs: (Batch, N_refls, Channels)

        # 1. Local Processing
        x = self.conv1(inputs)
        x = self.conv2(x)

        # 2. Low-Rank Global Mixing
        if self.use_global:
            # Transpose to (Batch, Channels, N_refls) so Dense acts on reflections
            x_T = tf.transpose(x, perm=[0, 2, 1])

            # Factorized Dense: x @ U @ V
            # U reduces dim to 'rank', V expands back to 'N_refls'
            # Initialize V with zeros -> Identity map at start -> No bias
            x_low = self.mix_U(x_T)
            x_global = self.mix_V(x_low)

            # Transpose back
            x_global = tf.transpose(x_global, perm=[0, 2, 1])

            # Residual connection with norm
            x = self.ln(x + x_global)

        logits = self.conv_out(x)

        if self.kind == 'slopes':
            out = self.activation(logits) + self.min_derivative
            return tf.expand_dims(out, -2)
        elif self.kind == 'widths':
            probs = tf.nn.softmax(logits, axis=-1)
            range_ = self.spline_range - (self.bins * self.min_bin_width)
            out = self.min_bin_width + (range_ * probs)
            return tf.expand_dims(out, -2)
        elif self.kind == 'heights':
            probs = tf.nn.softmax(logits, axis=-1)
            range_ = self.spline_range - (self.bins * self.min_bin_height)
            out = self.min_bin_height + (range_ * probs)
            return tf.expand_dims(out, -2)

class ComplexCartesianFlow(SurrogatePosterior):
    """ Surrogate Posterior for Complex Structure Factors using a Mixture Base. """
    def __init__(self, loc, scale, depth=4, hidden_units=32, bins=16,
                 range_min=-10.0, range_max=10.0, inference_samples=1,
                 mixing_rank=None, name='ComplexCartesianFlow', **kwargs):

        super().__init__(distribution=None, name=name, **kwargs)

        self.n_refls = loc.shape[0]
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.spline_range = self.range_max - self.range_min
        self.inference_samples = inference_samples

        # Diagnostic: Check correlation complexity
        use_global = self.n_refls < 20000
        if use_global:
            print(f"\n[Global Mixing] Dense Triangular Map ENABLED for {self.n_refls} reflections.")
        else:
            print(f"\n[Global Mixing] DISABLED (N={self.n_refls} > 20000). Using Independent Flow.\n")

        # 4-Component Mixture Base
        n_components = 4
        self.mix_logits = tf.Variable(tf.zeros((self.n_refls, n_components)), name='mix_logits')

        r_init = scale
        angles = tf.constant([0.0, np.pi/2, np.pi, 3*np.pi/2], dtype=tf.float32)
        r_exp = tf.expand_dims(r_init, -1)
        ang_exp = tf.expand_dims(angles, 0)

        loc_real = r_exp * tf.cos(ang_exp)
        loc_imag = r_exp * tf.sin(ang_exp)
        base_loc_init = tf.stack([loc_real, loc_imag], axis=-1)
        self.base_loc = tf.Variable(base_loc_init, name='base_loc')

        scale_val = scale * 0.7
        scale_init = tf.stack([scale_val]*n_components, axis=1)
        scale_init = tf.stack([scale_init, scale_init], axis=-1)

        self.base_scale = tfp.util.TransformedVariable(
            scale_init, tfb.Softplus(), name='base_scale'
        )

        components_dist = tfd.MultivariateNormalDiag(
            loc=self.base_loc, scale_diag=self.base_scale
        )
        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.mix_logits),
            components_distribution=components_dist
        )

        bijectors = []
        self.conditioners = []

        for i in range(depth):
            # Pass n_refls to SplineNet to enable Global Mixing if N is small
            w_net = SplineNet(hidden_units, bins, 'widths', self.spline_range, n_refls=self.n_refls, mixing_rank=mixing_rank, name=f'w_{i}')
            h_net = SplineNet(hidden_units, bins, 'heights', self.spline_range, n_refls=self.n_refls, mixing_rank=mixing_rank, name=f'h_{i}')
            s_net = SplineNet(hidden_units, bins, 'slopes', self.spline_range, n_refls=self.n_refls, mixing_rank=mixing_rank, name=f's_{i}')
            self.conditioners.append((w_net, h_net, s_net))

            bijectors.append(tfb.RealNVP(
                num_masked=1,
                bijector_fn=lambda x, ou, idx=i: tfb.RationalQuadraticSpline(
                    bin_widths=self.conditioners[idx][0](x),
                    bin_heights=self.conditioners[idx][1](x),
                    knot_slopes=self.conditioners[idx][2](x),
                    range_min=self.range_min
                )
            ))
            bijectors.append(tfb.Permute(permutation=[1, 0]))

        chain = tfb.Chain(list(reversed(bijectors)))
        self.distribution = tfd.TransformedDistribution(distribution=base_dist, bijector=chain)

    @property
    def parameters(self):
        return {}

    def parameter_properties(self, dtype=tf.float32, num_classes=None):
        return {}

    def sample(self, n_samples=None):
        if n_samples is None: n_samples = self.inference_samples
        z = self.distribution.sample(n_samples)
        return tf.complex(z[..., 0], z[..., 1])

    def log_prob(self, z_complex):
        real = tf.math.real(z_complex)
        imag = tf.math.imag(z_complex)
        z_stacked = tf.stack([real, imag], axis=-1)
        lp = self.distribution.log_prob(z_stacked)
        return tf.reduce_sum(lp, axis=-1)

    def mean(self):
        z = self.sample(self.inference_samples)
        return tf.reduce_mean(z, axis=0)

    def stddev(self):
        z = self.sample(self.inference_samples)
        return tf.math.reduce_std(tf.abs(z), axis=0)

    def mean_intensity(self):
        z = self.sample(self.inference_samples)
        return tf.reduce_mean(tf.square(tf.abs(z)), axis=0)

    def moment_4_intensity(self):
        z = self.sample(self.inference_samples)
        return tf.reduce_mean(tf.pow(tf.abs(z), 4), axis=0)

class ParticleSurrogate(SurrogatePosterior):
    """
    A simple surrogate that holds explicit structure factor instances (particles).
    Intended for use with SGLD or MAP estimation.
    """
    def __init__(self, n_refls, n_particles=1, initial_loc=None, initial_scale=1.0, name='ParticleSurrogate', **kwargs):
        super().__init__(distribution=None, name=name, **kwargs)
        self.n_refls = n_refls
        self.n_particles = n_particles

        if initial_loc is None:
            init_val = tf.random.normal((n_particles, n_refls, 2), stddev=initial_scale)
        else:
            if initial_loc.dtype.is_complex:
                loc = tf.stack([tf.math.real(initial_loc), tf.math.imag(initial_loc)], axis=-1)
            else:
                loc = initial_loc
            init_val = loc + tf.random.normal((n_particles, n_refls, 2), stddev=initial_scale * 0.1)

        self.particles = tf.Variable(init_val, name='particles', dtype=tf.float32)

    def sample(self, n_samples=None):
        z = tf.complex(self.particles[..., 0], self.particles[..., 1])
        return z

    def log_prob(self, z):
        return tf.reduce_sum(tf.zeros_like(tf.math.real(z)), axis=-1)

    @property
    def parameters(self):
        return {}

    def parameter_properties(self, dtype=tf.float32, num_classes=None):
        return {}

    def mean(self):
        z = self.sample()
        return tf.reduce_mean(z, axis=0)

    def stddev(self):
        z = self.sample()
        return tf.math.reduce_std(tf.abs(z), axis=0)

    def mean_intensity(self):
        z = self.sample()
        return tf.reduce_mean(tf.square(tf.abs(z)), axis=0)

    def moment_4_intensity(self):
        z = self.sample()
        return tf.reduce_mean(tf.pow(tf.abs(z), 4), axis=0)
