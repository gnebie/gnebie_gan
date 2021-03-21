import tensorflow as tf
import os
import logging
logger = logging.getLogger("gnebie_gan")

from .abstract_save_model import AbstractSaveModel

class TensorflowSaveModel(AbstractSaveModel):
    def __init__(self, flags,  name):
        super().__init__(flags, name)
        self.checkpoint_prefix = os.path.join(self.checkpoint_path, "ckpt")
        
    def create_checkpoint(self, generator_optimizer, discriminator_optimizer, generator, discriminator):
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

    def restore_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_path))
