import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from careless.models.likelihoods.laue import LaueBase, ConvolvedLikelihood

class BackgroundMLP(tfk.layers.Layer):
    """A simple MLP to predict a strictly positive background from metadata."""
    def __init__(self, **kwargs):
        # Hardcode the architecture here to keep the signature clean
        n_layers = 3
        width = 32
        leakiness = 0.01
        
        super().__init__(**kwargs)
        layers = []
        for _ in range(n_layers):
            layers.append(tfk.layers.Dense(
                width, 
                activation=tfk.layers.LeakyReLU(leakiness) if leakiness else 'relu'
            ))
        
        layers.append(tfk.layers.Dense(1, activation='linear'))
        layers.append(tfk.layers.Lambda(lambda x: tfb.Softplus()(x)))
        
        self.network = tfk.Sequential(layers)

    def call(self, metadata):
        return tf.squeeze(self.network(metadata), axis=-1)

class BackgroundNormalLikelihood(LaueBase):
    """
    A Custom Laue Likelihood that learns an additive background term B.
    I_pred = (Sigma * |F|^2) + B(metadata)
    """
    def __init__(self, *args, **kwargs):
        # We accept *args and **kwargs exactly as LaueBase does
        super().__init__(*args, **kwargs)
        
        # Instantiate the network during construction with no positional args
        self.background_network = BackgroundMLP()

    def dist(self, inputs):
        I_obs = self.get_intensities(inputs)
        sigma = self.get_uncertainties(inputs)
        I_obs = tf.squeeze(I_obs)
        sigma = tf.squeeze(sigma)
        
        metadata = self.get_metadata(inputs)
        
        # Call the already-instantiated network to get B
        B = self.background_network(metadata)
        
        # Shift the observation by B to evaluate:
        # P(I_obs - B | Sigma * |F|^2, sigma) 
        # which is mathematically P(I_obs | Sigma * |F|^2 + B, sigma)
        loc = I_obs - B
        
        return tfd.Normal(loc, sigma)

    def call(self, inputs):
        # The standard Laue Base wrapper
        harmonic_id = self.get_harmonic_id(inputs)
        likelihood = self.dist(inputs)
        return ConvolvedLikelihood(likelihood, harmonic_id)
