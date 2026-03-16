import tensorflow as tf
import tf_keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from careless.models.likelihoods.laue import LaueBase, ConvolvedLikelihood

class BackgroundMLP(tfk.layers.Layer):
    """A simple MLP to predict a strictly positive background from metadata."""
    def __init__(self, n_layers=3, width=32, leakiness=0.01):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(tfk.layers.Dense(
                width, 
                activation=tfk.layers.LeakyReLU(leakiness) if leakiness else 'relu'
            ))
        
        # Final layer predicts a single value. 
        # We use a Softplus bijector to ensure the predicted background is strictly positive.
        layers.append(tfk.layers.Dense(1, activation='linear'))
        layers.append(tfk.layers.Lambda(lambda x: tfb.Softplus()(x)))
        
        self.network = tfk.Sequential(layers)

    def call(self, metadata):
        # Squeeze the output so it broadcasts cleanly with the 1D intensities
        return tf.squeeze(self.network(metadata), axis=-1)

class BackgroundNormalLikelihood(LaueBase):
    """
    A Custom Laue Likelihood that learns an additive background term B.
    I_pred = (Sigma * |F|^2) + B(metadata)
    """
    def __init__(self, n_layers=3, width=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_network = BackgroundMLP(n_layers, width)

    def dist(self, inputs):
        # 1. Extract the observed raw total intensity and its uncertainty
        I_obs = self.get_intensities(inputs)
        sigma = self.get_uncertainties(inputs)
        I_obs = tf.squeeze(I_obs)
        sigma = tf.squeeze(sigma)
        
        # 2. Extract the metadata (x, y, lambda, etc.)
        metadata = self.get_metadata(inputs)
        
        # 3. Predict the background B for every reflection
        B = self.background_network(metadata)
        
        # 4. Return a Distribution where the LOCATION is shifted by B
        # In careless, the variational model will pass (Sigma * |F|^2) into this
        # distribution's log_prob function. We want to evaluate:
        # P(I_obs | Sigma * |F|^2 + B, sigma)
        # We achieve this by defining the likelihood centered on I_obs - B
        
        loc = I_obs - B
        
        # We evaluate P(loc | Sigma * |F|^2, sigma). 
        # Mathematically this is identical to P(I_obs | Sigma * |F|^2 + B, sigma)
        return tfd.Normal(loc, sigma)

    def call(self, inputs):
        # Standard Laue convolution wrap (handles the harmonic overlap)
        harmonic_id = self.get_harmonic_id(inputs)
        likelihood = self.dist(inputs)
        return ConvolvedLikelihood(likelihood, harmonic_id)
