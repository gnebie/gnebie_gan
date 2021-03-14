
from keras.datasets.mnist import load_data
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from numpy import ones

from .abstract_data_manager import AbstractDataManager

import logging
logger = logging.getLogger("gnebie_gan")


class MnistData(AbstractDataManager):
    def __init__(self, flags):
        self.flags = flags
        self.dataset = self.load_real_sample()

    def get_dataset(self):
        return load_data()

    def load_real_sample(self):
        return self.load_real_sample_labeled(8)

    def generate_real_samples(self, n_samples):
        return self.generate_real_samples_labeled(n_samples)


class SubFolderData(AbstractDataManager):
    """
    get all the files of the folder, labeled by subfolder 
    """
    def __init__(self, flags):
        self.flags = flags
        self.dataset_path = '/home/gnebie/Downloads/anime-faces/clean'
        self.dataset = self.load_real_sample()

    def get_dataset(self):
        return []

    def load_real_sample(self):
        label = 8
        return self.load_real_sample_labeled(label)

    def generate_real_samples(self, n_samples):
        return self.generate_real_samples_labeled(n_samples)




class FolderData(AbstractDataManager):
    """
    get all the files of the folder 
    """
    def __init__(self, flags):
        self.flags = flags
        self.dataset = self.load_real_sample()

    def get_dataset(self):
        return []

    def load_real_sample(self):
        return self.load_real_sample_no_label()

    def generate_real_samples(self, n_samples):
        self.generate_real_samples_no_label(n_samples)
