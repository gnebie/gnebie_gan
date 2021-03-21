import os

import abc
import logging
logger = logging.getLogger("gnebie_gan")
import tensorflow as tf

# https://docs.python.org/fr/3/library/abc.html

class AbstractGanModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, flags):
        # save variables functions
        self.basename = None
        self.name = "model"
        self.flags = flags
        self.basename = flags.out
        self.latent_dim = flags.latent_dim

    @abc.abstractmethod
    def run_train_step(self, *args): 
        raise NotImplementedError('users must define to use this base class')

    @abc.abstractmethod
    def create_new_model(self):
        raise NotImplementedError('users must define to use this base class')

    @abc.abstractmethod
    def save_model(self): 
        raise NotImplementedError('users must define to use this base class')

    @abc.abstractmethod
    def save_last_model(self): 
        raise NotImplementedError('users must define to use this base class')

    @abc.abstractmethod
    def restore_last_model(self):
        raise NotImplementedError('users must define to use this base class')

    @abc.abstractmethod
    def print_summary(self): 
        raise NotImplementedError('users must define to use this base class')


    def retrieve_or_create_model(self):
        if self.flags.new:
            self.create_new_model()
            logger.info("Create a smilly new model")
            return
        try:
            self.restore_last_model()
        except Exception as e:
            logger.exception("Fail to resore the model")
            self.create_new_model()
            logger.info("No restore model found, recreate a model")


