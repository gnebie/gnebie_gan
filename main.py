import os

import logging
logger = logging.getLogger("gnebie_gan")

from model.model import Model
from set_up_program.get_json import get_json
from set_up_program.parse_arguments import parse_arguments
from set_up_program.setup_log import setup_log


from tools.save_data import SaveData
from model.model import Model
from model.simple_gan import SimpleGan
from data.mnist_data import MnistData

def simple_mnist_model(flags):
    logger.info("Create simple mnist gan model.")

    gan = SimpleGan(flags)
    datas = MnistData(flags)
    save_data = SaveData(flags, gan)

    model = Model(flags, gan, datas, save_data)
    return model


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
    model = simple_mnist_model(flags)

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