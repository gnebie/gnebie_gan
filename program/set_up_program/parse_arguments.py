import argparse

# https://docs.python.org/3/library/argparse.html

# 

def log_arguments(parser):
    parser.add_arg("-v", "--verbosity", default=0, help="increase output verbosity", action="count")
    parser.add_arg("-q", "--quiet", help="do not print verbosity", action="store_true")
    parser.add_arg("--loglevel", type=str, default="", help="file log level")
    parser.add_arg("--logfile", type=str, default= "filelogs.log", help="file log path/name")

def global_arguments(parser):
    parser.add_arg("-o", "--out", type=str, default= "./out", help="folder to send the data")
    parser.add_arg("-t", "--train", help="train the model", action="store_true")
    parser.add_arg("--new", help="create a new model", action="store_true")

def unused(self, parser):
    parser.add_arg('--cuda'  , action='store_true', help='enables cuda')
    parser.add_arg('--ngpu'  , type=int, default=1, help='number of GPUs to use')


def arguments(parser):
    log_arguments(parser)
    global_arguments(parser)
    parser.add_arg("-f", "--flag", help="store flag true", action="store_true")
    parser.add_arg("--latent_dim", type=int, default=50, help="the generator latent number")
    parser.add_arg("--samples_nbr", type=int, default=100, help="the samples number to be create")
    parser.add_arg("--dataset", type=str, default="dataset", help="the dataset path")
    parser.add_arg('--batch_size', type=int, default=128, help="the batch size")
    parser.add_arg('--epochs', type=int, default=10, help="the epoch numbre.")


def to_add(parser):
    # to add 


    parser.add_arg('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_arg('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_arg('--ngf', type=int, default=64)
    parser.add_arg('--ndf', type=int, default=64)
    parser.add_arg('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_arg('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_arg('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_arg('--model', type=int, default=1, help='1 for dcgan, 2 for illustrationGAN-like-GAN')
    parser.add_arg('--d_labelSmooth', type=float, default=0, help='for D, use soft label "1-labelSmooth" for real samples')
    parser.add_arg('--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D')
    parser.add_arg('--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G')
    parser.add_arg('--binary', action='store_true', help='z from bernoulli distribution, with prob=0.5')



def parse_arguments():
    # parser = argparse.ArgumentParser(prog='gnebie-GAN')
    parser = ConfigParseArg()
    arguments(parser)
    parsed = parser.parse_args()
    return parsed


class LegacyParser(object):
    def __init__(self, prog='gnebie-GAN'):
        self.parser = argparse.ArgumentParser(prog=prog)

    def add_arg(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)    

    def parse_args(self):
        return self.parser.parse_args()



import configargparse

class ConfigParseArg(object):
    def __init__(self, prog='gnebie-GAN'):
        parser = configargparse.ArgParser(default_config_files=['/etc/app/conf.d/*.conf', '~/.my_settings'])
        parser.add('-c', '--my-config', is_config_file=True, help='config file path')
        self.parser = parser

    def add_arg(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)    

    def parse_args(self):
        return self.parser.parse_args()
