

import abc

from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from numpy import ones

import logging
logger = logging.getLogger("gnebie_gan")


class AbstractDataManager(object):
    __metaclass__ = abc.ABCMeta
   
    def generate_real_samples_no_label(self, n_samples):
        # choose random instances
        ix = randint(0, self.dataset.shape[0], n_samples)
        # select images
        X = self.dataset[ix]
        return X

    def generate_real_samples_labeled(self, n_samples):
        # choose random instances
        ix = randint(0, self.dataset.shape[0], n_samples)
        # select images
        X = self.dataset[ix]
        # generate class labels
        y = ones((n_samples, 1))
        return X, y


    def load_real_sample_labeled(self, label):
        # load dataset
        (train_images, train_labels), (_, _) = self.get_dataset()
        # expand to 3d, e.g. add channels
        train_images = expand_dims(train_images, axis=-1)
        # select all of the examples for a given class
        selected_ix = train_labels == label
        train_images_selected = train_images[selected_ix]
        # convert from ints to floats
        train_images_selected = train_images_selected.astype('float32')
        # scale from [0,255] to [-1,1]
        train_images_selected = (train_images_selected - 127.5) / 127.5
        return train_images_selected

    def load_real_sample_no_label(self):
        # load dataset
        (train_images), (_) = self.get_dataset()
        # expand to 3d, e.g. add channels
        train_images = expand_dims(train_images, axis=-1)
        # convert from ints to floats
        train_images = train_images.astype('float32')
        # scale from [0,255] to [-1,1]
        train_images_scaled = (train_images - 127.5) / 127.5
        return train_images_scaled

    def get_dataset(self):
        raise NotImplementedError('users must define this base class')


    def load_real_sample(self):
        raise NotImplementedError('users must define this base class')

    def generate_real_samples(self, n_samples):
        raise NotImplementedError('users must define this base class')
