
from numpy import zeros
from numpy.random import randn
from numpy import ones
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal


from .abstract_model import AbstractGanModel

import logging
logger = logging.getLogger("gnebie_gan")


class SimpleGan(AbstractGanModel):
    def __init__(self, flags):
        super().__init__()
        self.flags = flags
        self.latent_dim = flags.latent_dim
        self.basename = flags.out
        self.retrieve_or_create_model()

    def create_new_model(self):
        self.create_generator()
        self.create_discriminator()
        self.conbinate_model()

    def conbinate_model(self):
        # define the combined generator and discriminator model, for updating the generator
        # make weights in the discriminator not trainable
        self.discriminator.trainable = False
        # connect them
        self._conbinate_model = Sequential()
        # add generator
        self._conbinate_model.add(self.generator)
        # add the discriminator
        self._conbinate_model.add(self.discriminator)
        # compile model
        opt = self.optimizer
        lost_function = self.lost_function()
        self._conbinate_model.compile(loss=lost_function, optimizer=opt)
        return self._conbinate_model

    @property
    def optimizer(self):
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
        """
        tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam', **kwargs
        )
        """
        return Adam(lr=0.0002, beta_1=0.5)

    def lost_function(self):
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/binary_crossentropy
        """
        tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=0
        )
        """
        return "binary_crossentropy"
    
    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def gan_model(self):
        return self._conbinate_model

    @property
    def combine_model(self):
        return self._conbinate_model

    @generator.setter
    def generator(self, generator):
        self._generator = generator

    @discriminator.setter
    def discriminator(self, discriminator):
        self._discriminator = discriminator

    @combine_model.setter
    def combine_model(self, combine_model):
        self._conbinate_model = combine_model

    def create_generator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # define model
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, kernel_initializer=init, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        # output 28x28x1
        model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
        self._generator = model
        
    def create_discriminator(self, in_shape=(28,28,1)):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # define model
        model = Sequential()
        # downsample to 14x14
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample to 7x7
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        self._discriminator = model

    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        x_input = generate_latent_points(self.latent_dim, n_samples)
        # predict outputs
        X = self.generator.predict(x_input)
        # create class labels
        y = zeros((n_samples, 1))
        return X, y

    def generate_fake_samples_images(self, n_samples):
        x, _ = self.generate_fake_samples(n_samples)
        return x

    def update_discriminator_real(self, Xy_real):
        X_real, y_real = Xy_real
        return self._discriminator.train_on_batch(X_real, y_real)

    def update_discriminator_fake(self, n_samples):
        # generate 'fake' examples
        X_fake, y_fake = self.generate_fake_samples(n_samples)

        return self._discriminator.train_on_batch(X_fake, y_fake)


    def calculate_generate_lost(self, n_batch):
        X_gan = generate_latent_points(self.latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
        g_loss = self.gan_model.train_on_batch(X_gan, y_gan)
        return g_loss



def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 





