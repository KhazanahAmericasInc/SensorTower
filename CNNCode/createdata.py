import pickle
import sys
import random
import os
from PIL import Image
import numpy as np
import argparse

def main(args):
    imagedir = os.path.join(os.getcwd(), args.input_directory)
    
    if not os.path.isdir(imagedir):
        sys.exit("Image directory does not exist: {}".format(imagedir))

    imagelistfile = os.path.join(os.getcwd(), args.input_list_file)

    if not os.path.isfile(imagelistfile):
        sys.exit("Image list file does not exist: {}".format(imagelistfile))

    imagelist = None
    with open(imagelistfile, 'r') as infile:
        imagelist = infile.readlines()    

    imagecount = len(imagelist)
    if (args.shuffle):
        random.shuffle(imagelist)
    data = {}
    data['features'] = np.ndarray(shape=(imagecount, args.img_height, args.img_width, args.num_channel), dtype=np.uint8)
    data['label'] = np.ndarray(shape=(imagecount, ), dtype = np.uint8)
   
    for count, img in enumerate(imagelist):
        label = int(img[0]) - 1 
        data['label'][count] = label
        if args.num_channel == 1:
            gray = np.asarray(Image.open(os.path.join(imagedir, img.strip())).convert('L'))
            data['features'][count] = gray[:, :, np.newaxis]
        else:
            data['features'][count] = np.asarray(Image.open(os.path.join(imagedir, img.strip())).convert('RGB'))

    print ("Creating {}".format(args.out_file))
    pickle.dump(data, open(args.out_file, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_list_file", default="resized/imagelist.csv", help="path to input image list file")
    parser.add_argument("--input_directory", default="resized", help="name of directory that contains images")
    parser.add_argument("--out_file", default="data.p", help="name of output file containing image data")
    parser.add_argument("--img_width", default = 64, type = int, help="width of images in pixels")
    parser.add_argument("--img_height", default = 64, type = int, help="height of images in pixels")
    parser.add_argument("--num_channel", choices = [1, 3], default = 3, type = int, help="number of channels in image")
    parser.add_argument("--shuffle", action="store_true", help="option to shuffle data")
    args = parser.parse_args()
    main(args)
