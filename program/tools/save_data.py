from matplotlib import pyplot
import os
import tensorflow as tf
import logging
logger = logging.getLogger("gnebie_gan")

from .image import create_gif


class SaveData(object):
    def __init__(self, flags, gan_model):
        self.gan_model = gan_model
        self.flags = flags 
        self.basename = flags.out
        # self.n_samples = flags.samples_nbr
        # change the sample sizes
        self.samples_size = 8
        self.samples_path = os.path.join(self.basename, "samples")
        os.makedirs(self.samples_path, exist_ok=True)
        # prepare lists for storing stats each iteration
        self.d1_hist = list()
        self.d2_hist = list()
        self.g_hist = list()
        self.a1_hist = list()
        self.a2_hist = list()
        # self.seed = tf.random.normal([num_examples_to_generate, noise_dim])


    def record_history(self, step, d_loss1, d_loss2, g_loss, d_acc1, d_acc2):
        # summarize loss on this batch
        logger.notice('>%d, d1=%.3f, d2=%.3f g=%.3f, a1=%d, a2=%d' %
            (step+1, d_loss1, d_loss2, g_loss, int(100*d_acc1), int(100*d_acc2)))
        self.d1_hist.append(d_loss1)
        self.d2_hist.append(d_loss2)
        self.g_hist.append(g_loss)
        self.a1_hist.append(d_acc1)
        self.a2_hist.append(d_acc2)
    
    def plot_history(self):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.d1_hist, label='d-real')
        pyplot.plot(self.d2_hist, label='d-fake')
        pyplot.plot(self.g_hist, label='gen')
        pyplot.legend()
        # plot discriminator accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.a1_hist, label='acc-real')
        pyplot.plot(self.a2_hist, label='acc-fake')
        pyplot.legend()
        # save plot to file
        pyplot.savefig(self.basename + '/plot_line_plot_loss.png')
        pyplot.close()


    def save_generated_sample(self, step):
        sample_fullsize = self.samples_size * self.samples_size
        X = self.gan_model.generate_fake_samples_images(sample_fullsize)
        # scale from [-1,1] to [0,1]
        X = (X + 1) / 2.0
        # plot images
        for i in range(sample_fullsize):
            # define subplot
            pyplot.subplot(self.samples_size, self.samples_size, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
        # save plot to file
        pyplot.savefig(os.path.join(self.samples_path, 'generated_plot_%03d.png' % (step)))
        pyplot.close()


    def create_gif(self):
        create_gif(os.path.join(self.basename, "result.gif"), os.path.join(self.samples_path, "*.png"))


 
