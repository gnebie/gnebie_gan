import logging
logger = logging.getLogger("gnebie_gan")

from .abstract_save_model import AbstractSaveModel

class ExternalSaveModel(AbstractSaveModel):
    def __init__(self, flags):
        super().__init__(flags)

    def save_checkpoint(self, *args, **kwars):
        self.save_model(self, *args)

    def save_last_checkpoint(self, *args, **kwars):
        self.save_last_model(self, *args)

    def restore_checkpoint(self):
        self.restore_last_model(self, *args)

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

    def get_models(self, generator, discriminator, combine_model):
        self.generator = generator
        self.discriminator = discriminator
        self.combine_model = combine_model

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

