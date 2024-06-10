import tensorflow as tf
import numpy as np


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, image_shape, img_channels=1, load_model=False):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(image_shape, image_shape, img_channels)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=int(image_shape/4)*int(image_shape/4)*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(int(image_shape/4), int(image_shape/4), 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=1, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=img_channels, kernel_size=3, strides=1, padding='same'),
        ]
    )

    print("Encoder summary:\n") 
    self.encoder.summary()
    print("Decoder summary:\n")
    self.decoder.summary()

    if load_model:
      # Load the previously saved weights
      self.encoder.load_weights("training_1/cp-encoder-0100.weights.h5")
      self.decoder.load_weights("training_1/cp-decoder-0100.weights.h5")

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits


  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)


  def inference_batch_images(self, input_images):
    mean, logvar = self.encode(input_images)
    z = self.reparameterize(mean, logvar)
    predictions = self.sample(z)
    return predictions


  def inference_image(self, input_image):
    input_image = tf.expand_dims(input_image, axis=0)
    mean, logvar = self.encode(input_image)
    z = self.reparameterize(mean, logvar)
    predictions = self.sample(z)
    return predictions[0, :, :, :]


  def log_normal_pdf(self, sample, mean, logvar, raxis=1):
    '''
    This is the logarithm of the probability according to a normal distribution. 
    I.e. log(p(x)) where p is a normal/Gaussian distribution. The naming is a little confusing though.
    '''
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


  def compute_loss(self, x):
    '''
    VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood.
    In practice, optimize the single sample Monte Carlo estimate of this expectation:
            log p(x|z) + log p(z) - log q(z|x)  , where z is sampled from q(z|x)
    '''
    mean, logvar = self.encode(x)
    z = self.reparameterize(mean, logvar)
    ## TODO: FLOAT16? OR FLOAT32?
    x_logit = tf.cast(self.decode(z), tf.float32)
    cross_ent = tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x), tf.float32)
    logpx_z = tf.cast(-tf.reduce_sum(cross_ent, axis=[1, 2, 3]), tf.float32)
    logpz = tf.cast(self.log_normal_pdf(z, 0., 0.), tf.float32)
    logqz_x = tf.cast(self.log_normal_pdf(z, mean, logvar), tf.float32)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


  @tf.function
  def train_step(self, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
      loss = self.compute_loss(x)
    gradients = tape.gradient(loss, self.trainable_variables)
    optimizer.apply_gradients(zip(gradients, self.trainable_variables))