import tensorflow as tf
import logging
logger = logging.getLogger("gnebie_gan")

from .abstract_save_model import AbstractSaveModel

class TensorflowSaveModel(AbstractSaveModel):
    def __init__(self, flags, gan_model):
        self.gan_model = gan_model
        self.flags = flags 
        self.basename = flags.out
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        
    def create_checkpoint(self, generator_optimizer, discriminator_optimizer, generator, discriminator)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    def save_checkpoint(self):
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

    def restore_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
