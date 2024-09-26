import cv2
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Get images.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--height', type=int, default=0, help='the height of the new imager.')
    parser.add_argument('--width', type=int, default=0, help='the width of the new image.')
    group.add_argument('--resize-percent', type=int, default=0, choices=range(10, 400), help='the resize of the new image.')
    parser.add_argument('-i', '--input-folder', type=str, nargs="+", required=True, help='the input folder.')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='the output folder.')

    args = parser.parse_args()
    return args

def prepare_width_height_percent(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return width, height

def prepare_and_launch():
    args = get_args()
    height = 0
    width = 0
    if args.height > 0:
        height = args.height 
        if args.width > 0:
            width =  args.width 
        else:
            width = height
    for folder in args.input_folder:
        if os.path.isdir(folder):
            launch_images(args, folder, folder, "", width, height)
 

def launch_images(args, origin, folder, subfolder, width, height):
    for file_found in os.listdir(folder):
        print("folder found {}".format(folder))
        if os.path.isfile(os.path.join(folder, file_found)):
            print("filer found {}".format(file_found))
            # test the extention here
            ext = os.path.splitext(file_found)[-1].lower()
            if ext in [".png", ".jpg"]:
                transform_images(args, origin, file_found, subfolder, width, height)
        elif os.path.isdir(os.path.join(folder, file_found)):
            new_folder = os.path.join(folder, file_found)
            new_subfolder = os.path.join(subfolder, file_found)
            launch_images(args, origin, new_folder, new_subfolder, width, height)


def transform_images(args, origin, file_name, file_path, width, height):
    if args.resize_percent > 0:
        width, height = prepare_width_height_percent(img, args.resize_percent) 
    
    print("image found {}".format(file_name))
    file_p = os.path.join(origin, file_path, file_name)
    img = cv2.imread(file_p, cv2.IMREAD_UNCHANGED)

    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    os.makedirs(os.path.join(args.output_folder, file_path), exist_ok=True)
    filename = os.path.join(args.output_folder, file_path, file_name) 
    cv2.imwrite(filename, resized) 

def main():
    prepare_and_launch()
  
if __name__ == '__main__':
    main()

