import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import randn
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow as tf

from IPython import display

import logging
logger = logging.getLogger("gnebie_gan")
from .save_model.tensorflow_save_model import TensorflowSaveModel
from .abstract_model import AbstractGanModel

import logging
logger = logging.getLogger("gnebie_gan")

class tensorflowDCGan(AbstractGanModel):
    def __init__(self, flags):
        super().__init__(flags)
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.save = TensorflowSaveModel(flags, self.name)
        self.retrieve_or_create_model()
        self.noise_dim = 100
        self.batch_size = flags.batch_size


    def create_new_model(self):
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.save.create_checkpoint(self.generator_optimizer, self.discriminator_optimizer, self.generator, self.discriminator)

    def create_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model
        
    def create_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self._cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def generator_loss(self, fake_output):
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/binary_crossentropy
        return self._cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def save_model():
        self.save.save_checkpoint()

    def restore_last_model():
        self.create_new_model()
        self.save.restore_checkpoint()

    def restore_model():
        self.save.restore_checkpoint()

    def generate_fake_samples_images(self, n_samples):
        return self.generator(tf.random.normal([1, 100]), training=False)
        image_nbr = generate_latent_points(self.latent_dim, n_samples)
        X, y = self.generator.predict(image_nbr)
        return X
        return [ self.generator(tf.random.normal([1, 100]), training=False) for i in range(image_nbr) ]


    @tf.function
    def train_steps(self, images, step):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def run_train_step(self, images, step):
        self.train_steps(images, step)

    def print_summary(self):
        self.generator.summary()
        self.discriminator.summary()

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 

