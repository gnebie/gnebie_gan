import time
import logging
logger = logging.getLogger("gnebie_gan")

class Model(object):
    def __init__(self, flags, gan, data, save):
        self.flags = flags
        self.gan_model = gan
        self.datas = data
        self.saves_tools = save
        # Print the model summary
        self.gan_model.print_summary()

    def train(self):
        step = 0
        for epoch in range(self.flags.epochs):
            self.datas.start_batch()
            start = time.time()
            logger.debug("epoch {} start at {}".format(epoch, start))
            for batch in self.datas.get_batches():
                logger.info("step : {}".format(step))
                self.gan_model.run_train_step(batch, step)
                step += 1
            logger.info("epoch %i in {} seconds".format(time.time() - start), epoch)
            self.saves_tools.save_generated_sample(step)
            self.gan_model.save_model(step)
        self.saves_tools.plot_history()
        self.saves_tools.create_gif()
        
        self.gan_model.save_last_model()

    def create(self):
        return None
