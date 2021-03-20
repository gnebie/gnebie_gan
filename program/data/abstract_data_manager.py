

import abc

from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from numpy import ones

import logging
logger = logging.getLogger("gnebie_gan")

# https://www.tensorflow.org/tutorials/load_data/images

class AbstractDataManager(object):
    __metaclass__ = abc.ABCMeta
   
    def __init__(self, flags):
        self.flags = flags
        self._dataset = self.get_dataset()

    @property
    def dataset(self):
        self._dataset

    def get_dataset(self):
        raise NotImplementedError('users must define this base class')

    def load_real_sample(self):
        raise NotImplementedError('users must define this base class')

    def generate_real_samples(self, n_samples):
        raise NotImplementedError('users must define this base class')

#region new elems
    # new
    def get_batches(self):
        raise NotImplementedError('users must define this base class')

    def start_batch(self):
        pass
#endregion











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







    def resharp_images(self, images, width, height):
        return images.reshape(images.shape[0], 28, 28, 1).astype('float32')

    def normalize_images(self, images):
        # Normalize the images to [-1, 1]
        return (images - 127.5) / 127.5 

    def normalize_images_0_1(self, image_label_batch):
        # Normalize the images to [0, 1]
        from tensorflow.keras import layers

        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        normalized_ds = image_label_batch.map(lambda x, y: (normalization_layer(x), y))

        return normalized_ds 


    def get_dataset_from_disk(self, data_dir):
        # https://www.tensorflow.org/tutorials/load_data/images
        batch_size = 32
        img_height = 180
        img_width = 180
        color_mode = "rgb"

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            color_mode=color_mode,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            color_mode=color_mode,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        # https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        return train_ds, val_ds 

    def next_batch(num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

