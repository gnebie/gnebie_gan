import PIL
import imageio
import glob


def display_image(path):
  return PIL.Image.open(path)


def create_gif(anim_file='dcgan.gif', image_glob='image*.png'):
    # https://imageio.readthedocs.io/en/stable/format_gif-pil.html#gif-pil
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(image_glob)
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def random_crop(image, img_height, img_width, color_layers=3):
    cropped_image = tf.image.random_crop(
        image, size=[img_height, img_width, color_layers])
    return cropped_image


# load an image as an rgb numpy array
def load_image(filename):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	return pixels

"""
  resize : 
  required_size = (128, 128)
  return image.resize(required_size)
"""
