

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


    def run_train_step(self, epoch, bat, n_batch, half_batch, i):
        # get randomly selected 'real' samples
        X_real = self.datas.generate_real_samples(half_batch)

        # update discriminator model weights
        d_loss1, d_acc1 = self.gan_model.update_discriminator_real(X_real)

        # update discriminator model weights
        d_loss2, d_acc2 = self.gan_model.update_discriminator_fake(half_batch)

        # prepare points in latent space as input for the generator
        g_loss = self.gan_model.calculate_generate_lost(n_batch)
        
        # record history
        self.saves_tools.record_history(i, d_loss1, d_loss2, g_loss, d_acc1, d_acc2)

    def train(self, n_epochs=10, n_batch=128):
        # TODO: make the code before the loop variable

        # calculate the number of batches per epoch
        bat_per_epo = int(self.datas.dataset.shape[0] / n_batch)
 
        # calculate the total iterations based on batch and epoch
        n_steps = bat_per_epo * n_epochs
 
        # calculate the number of samples in half a batch
        half_batch = int(n_batch / 2)
        logger.info("bat_per_epo: %d, half_batch %d, n_steps : %d ", bat_per_epo, half_batch, n_steps)

        # TODO: end

        i = 0
        for epoch in range(n_epochs):
            logger.info("epoch nbr %i", epoch)

            for bat in range(bat_per_epo):
                self.run_train_step(epoch, bat, n_batch, half_batch, i)
                i += 1
            
            logger.info("bat nbr %i of epoch %i", bat, epoch)

            # evaluate the model performance every 'epoch'
            self.saves_tools.save_generated_sample(i)
            self.gan_model.save_model(i)

        self.saves_tools.plot_history()
        self.saves_tools.create_gif()
        
        self.gan_model.save_last_model()


    def create(self):
        return None
