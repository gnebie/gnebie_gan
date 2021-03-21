import os
import abc
import logging
logger = logging.getLogger("gnebie_gan")

class AbstractSaveModel(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, flags, name):
        # save variables functions
        self.flags = flags
        self.name = name
        self.basename = flags.out
        self.checkpoint_path = os.path.join(self.basename, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)

    @abc.abstractmethod
    def save_checkpoint(self, *args, **kwars):
        raise NotImplementedError('users must define to use this base class')

    @abc.abstractmethod
    def save_last_checkpoint(self, *args, **kwars):
        raise NotImplementedError('users must define to use this base class')

    @abc.abstractmethod
    def restore_checkpoint(self):
        raise NotImplementedError('users must define to use this base class')
