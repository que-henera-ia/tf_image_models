import tensorflow as tf
from tensorflow.keras import layers
from tf_utils.process_images import clip_0_1


class DCGAN():
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, image_shape, image_channels=1):
    super(DCGAN, self).__init__()
    self.latent_dim = latent_dim
    self.image_shape = image_shape
    self.image_channels = image_channels
    self.generator = self.make_generator_model()
    print("Generator summary:\n") 
    self.generator.summary()
    self.discriminator = self.make_discriminator_model()
    print("Discriminator summary:\n")
    self.discriminator.summary()

    # This method returns a helper function to compute cross entropy loss
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


  def make_generator_model(self):
    model = tf.keras.Sequential()
    model.add(layers.Dense(int(self.image_shape/4*self.image_shape/4)*256, use_bias=False, input_shape=(self.latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(self.image_shape/4), int(self.image_shape/4), 256)))
    assert model.output_shape == (None, int(self.image_shape/4), int(self.image_shape/4), 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(self.image_shape/4), int(self.image_shape/4), 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(self.image_shape/4), int(self.image_shape/4), 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(self.image_shape/2), int(self.image_shape/2), 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, int(self.image_shape/2), int(self.image_shape/2), 8)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(self.image_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, self.image_shape, self.image_shape, self.image_channels)

    return model

  def generator_loss(self, fake_output):
    return self.cross_entropy(tf.ones_like(fake_output), fake_output)

  def generate_images(self, input=None):
    if input is None:
      num_images_to_generate = 1
      input = tf.random.normal([num_images_to_generate, self.latent_dim])
    return clip_0_1(self.generator(input, training=False))

  def make_discriminator_model(self):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[self.image_shape, self.image_shape, self.image_channels]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

  def discriminator_loss(self, real_output, fake_output):
    real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  # def decode(self, z, apply_sigmoid=False):
  #   logits = self.decoder(z)
  #   if apply_sigmoid:
  #     probs = tf.sigmoid(logits)
  #     return probs
  #   return logits

  # Notice the use of `tf.function`
  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self, images, batch_size):
      noise = tf.random.normal([batch_size, self.latent_dim])

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = self.generator(noise, training=True)
        generated_images = clip_0_1(generated_images) #tanh output, clipping is needed

        real_output = self.discriminator(images, training=True)
        fake_output = self.discriminator(generated_images, training=True)

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)

      gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

      return gen_loss, disc_loss, generated_images