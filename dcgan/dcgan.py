import tensorflow as tf
from tensorflow.keras import layers
from tf_utils.process_images import clip_0_1
import numpy as np


class DCGAN(tf.keras.Model):
  """Deep Convolutional Generative Adversarial Network."""

  def __init__(self, model_name, latent_dim, image_shape, image_channels=1, checkpoint_path="training/", seed=None, seed_length=4):
    super(DCGAN, self).__init__()

    self.model_name = model_name
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

    self.generator_checkpoint_path = checkpoint_path+model_name+"-generator-"
    self.discriminator_checkpoint_path = checkpoint_path+model_name+"-discriminator-"

    self.seed_length = seed_length
    if seed is None:
      self.seed=tf.random.normal([self.seed_length, self.latent_dim])

    self.loss_names = ["Generator Loss", "Discriminator Loss"]


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

  @tf.function
  def generate_images(self, seed=None):
    if seed is None:
      seed = tf.random.normal([self.seed_length, self.latent_dim])
    return clip_0_1(self.generator(seed, training=False))

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

  def classify_image(self, image):
    return self.discriminator(image)

  def compute_loss(self, images):
    ## TODO: check this function. Is the same using batch_size that len(images) in the generation of noise?
    noise = tf.random.normal([len(images), self.latent_dim])

    generated_images = self.generator(noise, training=True)
    generated_images = clip_0_1(generated_images) #tanh output, clipping is needed

    real_output = self.discriminator(images, training=True)
    fake_output = self.discriminator(generated_images, training=True)

    gen_loss = self.generator_loss(fake_output)
    disc_loss = self.discriminator_loss(real_output, fake_output)

    return gen_loss, disc_loss

  # Notice the use of `tf.function`
  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self, images):

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_loss, disc_loss = self.compute_loss(images)

      ## TODO: the following functions must be inside the with? We are using gen_tape and disc_tape outside the with
      gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

      # return [gen_loss, disc_loss] # ["Generator Loss", "Discriminator Loss"]


  def save_weights(self, add_text=""):
    self.generator.save_weights(self.generator_checkpoint_path+add_text+".weights.h5")
    self.discriminator.save_weights(self.discriminator_checkpoint_path+add_text+".weights.h5")

  def load_weights(self, add_text=""):
    self.generator.load_weights(self.generator_checkpoint_path+add_text+".weights.h5")
    self.discriminator.load_weights(self.discriminator_checkpoint_path+add_text+".weights.h5")