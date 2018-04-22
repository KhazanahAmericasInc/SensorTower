import pickle
import sys
import random
import os
from PIL import Image
import numpy as np
import argparse

###
# This script takes a list of images and reads them all into a binary file. The created binary file essentially contains
# all the images. This is done for two reasons. First, by creating one file of all the images, it requires the following
# scripts to only need to open one file, speeding up the process. In addition, it saves an immutable snapshot of the data.
# if images are deleted or added to the folder or list, it would not interfere with the binary file, keeping training
# consistent.
###

# This function takes the input command arguments. It reads the file that lists the images and open all the images.
# It then saves the images to a single binary file.
def main(args):
    imagedir = os.path.join(os.getcwd(), args.input_directory)
    
    if not os.path.isdir(imagedir):
        sys.exit("Image directory does not exist: {}".format(imagedir))

    imagelistfile = os.path.join(os.getcwd(), args.input_list_file)

    if not os.path.isfile(imagelistfile):
        sys.exit("Image list file does not exist: {}".format(imagelistfile))

    # Read the image list file.
    imagelist = None
    with open(imagelistfile, 'r') as infile:
        imagelist = infile.readlines()    

    imagecount = len(imagelist)

    # Shuffle the list of images
    if (args.shuffle):
        random.shuffle(imagelist)
    data = {}
    data['features'] = np.ndarray(shape=(imagecount, args.img_size, args.img_size, args.num_channel), dtype=np.uint8)
    data['label'] = np.ndarray(shape=(imagecount, ), dtype = np.uint8)
   
    # Iterates through all the images and saves them to the binary file
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
    parser.add_argument("--img_size", default = 64, type = int, help="size of image in pixels (assumed square)")
    parser.add_argument("--num_channel", choices = [1, 3], default = 3, type = int, help="number of channels in image")
    parser.add_argument("--shuffle", action="store_true", help="option to shuffle data")
    args = parser.parse_args()
    main(args)
