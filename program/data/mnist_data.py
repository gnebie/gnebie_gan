
from keras.datasets.mnist import load_data
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from numpy import ones
import tensorflow as tf

from .abstract_data_manager import AbstractDataManager

import logging
logger = logging.getLogger("gnebie_gan")


class MnistData(AbstractDataManager):
    def __init__(self, flags):
        super().__init__(flags)

    def get_dataset(self):
        return load_data()

    def get_batches(self):
        return [ self.generate_real_samples(self.batch_size) for i in range(self._batches) ]
    def get_batch(self):
        if self._batches > 0:
            self._batches -= 1
            images = self.generate_real_samples(self.batch_size)
            return [images]
        return None

    def start_batch(self):
        dataset_len = len(self.dataset)
        logger.debug("dataset : {}  batch_size {}".format(dataset_len, self.batch_size))
        
        self._batches = int(dataset_len / self.batch_size)
        logger.debug("self._batches : {}  ".format(self._batches))

    def load_normalize_dataset(self):
        return self.load_normalize_dataset_labeled(8)

    def generate_real_samples(self, n_samples):
        return self.generate_real_samples_labeled(n_samples)


class TensorFlowMnistData(AbstractDataManager):
    def __init__(self, flags):
        super().__init__(flags)

    def get_dataset(self):
        return load_data()

    def load_normalize_dataset(self):
        (train_images, train_labels), (_, _) = self.get_dataset()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        BUFFER_SIZE = 60000
        BATCH_SIZE = self.flags.batch_size
        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return (train_images, train_labels)

    def get_batches(self):
        return self.train_dataset

    def start_batch(self):
        pass





class SubFolderData(AbstractDataManager):
    """
    get all the files of the folder, labeled by subfolder 
    """
    def __init__(self, flags):
        super().__init__(flags)
        self.flags = flags
        self.dataset_path = '/home/gnebie/Downloads/anime-faces/clean'
        self.dataset = self.load_normalize_dataset()
        self.batch 

    def get_dataset(self):
        return []

    def load_normalize_dataset(self):
        label = 8
        return self.load_normalize_dataset_labeled(label)

    def generate_real_samples(self, n_samples):
        return self.generate_real_samples_labeled(n_samples)




class FolderData(AbstractDataManager):
    """
    get all the files of the folder 
    """
    def __init__(self, flags):
        super().__init__(flags)
        self.flags = flags
        self.dataset = self.load_normalize_dataset()

    def get_dataset(self):
        return []

    def load_normalize_dataset(self):
        return self.load_normalize_dataset_no_label()

    def generate_real_samples(self, n_samples):
        self.generate_real_samples_no_label(n_samples)
