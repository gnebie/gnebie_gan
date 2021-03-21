import os

import logging
logger = logging.getLogger("gnebie_gan")

from model.model import Model
from set_up_program.get_json import get_json
from set_up_program.parse_arguments import parse_arguments
from set_up_program.setup_log import setup_log


from tools.save_data import SaveData
from model.gan_models.simple_gan import SimpleGan
from model.gan_models.tensorflow_simple_gan import tensorflowDCGan
from data.mnist_data import MnistData
from data.mnist_data import TensorFlowMnistData

def simple_mnist_model(flags):
    logger.info("Create simple mnist gan model.")

    datas = MnistData(flags)

    gan = SimpleGan(flags)
    save_data = SaveData(flags, gan)

    model = Model(flags, gan, datas, save_data)
    return model

def simple_tensorflow_mnist_model(flags):
    logger.info("Create simple mnist gan model.")

    datas = TensorFlowMnistData(flags)

    gan = tensorflowDCGan(flags)
    save_data = SaveData(flags, gan)

    model = Model(flags, gan, datas, save_data)
    return model


def create_model(flags):
    return simple_tensorflow_mnist_model(flags)
    return simple_mnist_model(flags)

def setup_info():
    # get parse infos
    flags = parse_arguments()

    # Create the results dir if needed
    os.makedirs(flags.out, exist_ok=True)

    # Create the logs
    setup_log(flags)
    
    return flags

def main():
    flags = setup_info()

    # Create the model
    logger.trace("Create model:")
    model = create_model(flags)
    
    # run the model
    if flags.train:
        try:
            logger.info(" ********* Train the model ********* ")
            model.train()
        except Exception:
            logger.exception("Error during the train found :")
    else:
        try:
            logger.info(" ********* Create samples using the pretrained model ********* ")
            model.create()
        except Exception:
            logger.exception("Error during the creation found : ")



if __name__ == '__main__':
    main()

"""

achitecture


main -
        model : run the model
        datasets : load datas
        gan_model : generator/discriminator model
                    model_save: save the model
        save_samples: save samples and infos  

"""