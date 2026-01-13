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

        # 2. Build the Distribution
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

        # Enforce positivity
        bijectors.append(tfb.Softplus())

        chain = tfb.Chain(bijectors)
        distribution = tfd.TransformedDistribution(distribution=base_dist, bijector=chain)

        # 3. Initialize superclass
        super().__init__(distribution, name=name, **kwargs)

        # 4. Assign variables to self so Keras tracks them
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
        if n_samples is None:
            n_samples = self.inference_samples
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
    A helper Keras Layer that predicts spline parameters (widths, heights, or slopes).
    """
    def __init__(self, hidden_units, bins, kind, spline_range, name=None):
        super().__init__(name=name)
        self.hidden_units = hidden_units
        self.bins = bins
        self.kind = kind
        self.spline_range = spline_range

        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_units, activation='relu')

        if kind == 'slopes':
            # Predict (bins - 1) interior slopes.
            self.dense_out = tf.keras.layers.Dense(bins - 1)
            self.activation = tf.keras.layers.Activation('softplus')
        else:
            # Predict 'bins' widths/heights
            self.dense_out = tf.keras.layers.Dense(bins)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        out = self.dense_out(x)

        if self.kind == 'slopes':
            out = self.activation(out)
            # Reshape to (Batch, 1, bins-1) for broadcasting
            return tf.expand_dims(out, 1)
        else:
            # Softmax to ensure sum is 1, then scale to total range
            probs = tf.nn.softmax(out, axis=-1)
            return tf.expand_dims(probs * self.spline_range, 1)

class ComplexCartesianFlow(SurrogatePosterior):
    """
    A Surrogate Posterior for Complex Structure Factors (Amplitude and Phase).
    Models the joint distribution of Real and Imaginary components using RQS.
    """
    def __init__(self, loc, scale, depth=4, hidden_units=32, bins=32,
                 range_min=-10.0, range_max=10.0, inference_samples=1,
                 name='ComplexCartesianFlow', **kwargs):

        # 1. Initialize Keras Model first (passing None for distribution)
        super().__init__(distribution=None, name=name, **kwargs)

        self.n_refls = loc.shape[0]
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.spline_range = self.range_max - self.range_min
        self.inference_samples = inference_samples

        # 2. Base Distribution: Independent Normal on Real/Imag parts
        base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(2 * self.n_refls),
            scale_diag=tf.ones(2 * self.n_refls)
        )

        bijectors = []

        # 3. Create Layers (tracked by self)
        self.conditioners = []
        for i in range(depth):
            # A. Permutation
            bijectors.append(tfb.Permute(permutation=np.random.permutation(2 * self.n_refls)))

            # B. Conditioners
            w_net = SplineNet(hidden_units, bins, 'widths', self.spline_range, name=f'w_{i}')
            h_net = SplineNet(hidden_units, bins, 'heights', self.spline_range, name=f'h_{i}')
            s_net = SplineNet(hidden_units, bins, 'slopes', self.spline_range, name=f's_{i}')
            self.conditioners.append((w_net, h_net, s_net))

            # C. RQS Coupling
            bijectors.append(tfb.RealNVP(
                num_masked=self.n_refls,
                bijector_fn=lambda x, ou, idx=i: tfb.RationalQuadraticSpline(
                    bin_widths=self.conditioners[idx][0](x),
                    bin_heights=self.conditioners[idx][1](x),
                    knot_slopes=self.conditioners[idx][2](x),
                    range_min=self.range_min
                )
            ))

        # 4. Final Scale/Shift to Physical Space
        physical_scale = tf.concat([scale, scale], axis=0)
        physical_loc = tf.concat([loc, loc], axis=0)
        bijectors.append(tfb.Shift(physical_loc)(tfb.Scale(physical_scale)))

        chain = tfb.Chain(list(reversed(bijectors)))

        # 5. Assign Distribution
        self.distribution = tfd.TransformedDistribution(distribution=base_dist, bijector=chain)

    @property
    def parameters(self):
        # Override to prevent exporting 'bijector' chain to MTZ
        return {}

    def parameter_properties(self, dtype=tf.float32, num_classes=None):
        # Override to prevent get_results from iterating over internal props
        return {}

    def sample(self, n_samples=None):
        if n_samples is None:
            n_samples = self.inference_samples

        flat_sample = self.distribution.sample(n_samples)
        z = tf.reshape(flat_sample, (n_samples, self.n_refls, 2))
        return tf.complex(z[..., 0], z[..., 1])

    def log_prob(self, z_complex):
        real = tf.math.real(z_complex)
        imag = tf.math.imag(z_complex)

        if len(z_complex.shape) == 1:
            flat_input = tf.concat([real, imag], axis=0)
        else:
            flat_input = tf.reshape(tf.stack([real, imag], axis=-1), (tf.shape(z_complex)[0], -1))

        return self.distribution.log_prob(flat_input)

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
