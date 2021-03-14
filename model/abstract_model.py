import os

import abc
import logging
logger = logging.getLogger("gnebie_gan")
import tensorflow as tf


class AbstractGanModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # save variables functions
        self.basename = None
        self.name = "model"

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError('users must define __str__ to use this base class')

    # @property
    # @abc.abstractmethod
    # def checkpoint_path(self):
    #     raise NotImplementedError('users must define __str__ to use this base class')

    @property
    @abc.abstractmethod
    def generator(self):
        raise NotImplementedError('users must define generator to use this base class')

    @property
    @abc.abstractmethod
    def discriminator(self):
        raise NotImplementedError('users must define discriminator to use this base class')

    @property
    @abc.abstractmethod
    def combine_model(self):
        raise NotImplementedError('users must define combine_model to use this base class')

    @generator.setter
    @abc.abstractmethod
    def generator(self):
        raise NotImplementedError('users must define generator to use this base class')

    @discriminator.setter
    @abc.abstractmethod
    def discriminator(self):
        raise NotImplementedError('users must define discriminator to use this base class')

    @combine_model.setter
    @abc.abstractmethod
    def combine_model(self):
        raise NotImplementedError('users must define combine_model to use this base class')

    @abc.abstractmethod
    def generate_fake_samples(self, n_samples):
        raise NotImplementedError('users must define this base class')

    @abc.abstractmethod
    def generate_fake_samples_images(self, n_sample):
        raise NotImplementedError('users must define this base class')

    @abc.abstractmethod
    def update_discriminator_real(self, Xy_real):
        raise NotImplementedError('users must define this base class')

    @abc.abstractmethod
    def update_discriminator_fake(self, n_sample):
        raise NotImplementedError('users must define this base class')

    @abc.abstractmethod
    def calculate_generate_lost(self, n_batch):
        raise NotImplementedError('users must define this base class')

    @abc.abstractmethod
    def create_new_model(self):
        raise NotImplementedError('users must define this base class')

    def print_summary(self):
        self.generator.summary()
        self.discriminator.summary()
        self.combine_model.summary()

    def retrieve_or_create_model(self):
        self.checkpoint_path = os.path.join(self.basename, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        if self.flags.new:
            self.create_new_model()
            logger.info("Create a smilly new model")
            return

        try:
            self.restore_last_model()
            logger.info("restore model from source")
        except Exception as e:
            logger.exception("Fail to resore the model")
            self.create_new_model()
            logger.info("No restore model found, recreate a model")

    # Basic save functions
    def get_model_path_from_step(self, step):
        model_name = '{}_{:03d}.h5'.format(self.name, step)
        model_path = os.path.join(self.checkpoint_path, model_name)  
        return model_path

    def get_model_path(self, subname):
        model_name = '{}_{}.h5'.format(self.name, subname)
        model_path = os.path.join(self.checkpoint_path, model_name)  
        return model_path

    def restore_model_weight(self, step):
        return self.combine_model.load_weights(self.get_model_path_from_step(step))

    def save_model_weight(self, step):
        self.combine_model.save_weights(self.get_model_path_from_step(step))

    def restore_model(self, step):
        return tf.keras.models.load_model(self.get_model_path_from_step(step))

    def save_model(self, step):
        self.combine_model.save(self.get_model_path_from_step(step))

    def restore_last_model(self):
        model_generator_path = self.get_model_path('generator')
        model_discriminator_path = self.get_model_path('discriminator')
        model_combine_path = self.get_model_path('combine')

        self.generator = tf.keras.models.load_model(model_generator_path)
        self.discriminator = tf.keras.models.load_model(model_discriminator_path)
        self.combine_model = tf.keras.models.load_model(model_combine_path)

    def save_last_model(self):
        model_generator_path = self.get_model_path('generator')
        model_discriminator_path = self.get_model_path('discriminator')
        model_combine_path = self.get_model_path('combine')

        self.generator.save(model_generator_path)
        self.discriminator.save(model_discriminator_path)
        self.combine_model.save(model_combine_path)

    def restore_last_weight(self):
        model_generator_path = self.get_model_path('generator_w')
        model_discriminator_path = self.get_model_path('discriminator_w')
        model_combine_path = self.get_model_path('combine_w')

        self.generator.load_weights(model_generator_path)
        self.discriminator.load_weights(model_discriminator_path)
        self.combine_model.load_weights(model_combine_path)

    def save_last_weights(self):
        model_generator_path = self.get_model_path('generator_w')
        model_discriminator_path = self.get_model_path('discriminator_w')
        model_combine_path = self.get_model_path('combine_w')

        self.generator.save(model_generator_path)
        self.discriminator.save(model_discriminator_path)
        self.combine_model.save(model_combine_path)

