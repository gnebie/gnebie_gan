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


    def train(self, n_epochs=10, n_batch=128):
        i = 0
        for epoch in range(n_epochs):
            self.datas.start_batch()
            start = time.time()
            logger.debug("epoch {} start at {}".format(epoch, start))
            for batch in self.datas.get_batches():
                self.gan_model.run_train_step(batch)
                i += 1
            logger.info("epoch %i in {} seconds".format(time.time() - start), bat, epoch)
            self.saves_tools.save_generated_sample(i)
            self.gan_model.save_model(i)
        self.saves_tools.plot_history()
        self.saves_tools.create_gif()
        
        self.gan_model.save_last_model()


    def create(self):
        return None
