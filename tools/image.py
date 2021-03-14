import PIL
import imageio
import glob


def display_image(path):
  return PIL.Image.open(path)


def create_gif(anim_file='dcgan.gif', image_glob='image*.png'):
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(image_glob)
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
