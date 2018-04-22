import argparse
import sys
import os
import cv2
import numpy as np

###
# This script takes a list of images and resizes it all to a specific size. It does this by resizing the larger
# dimension to the desired size, and padding the remaining pixels to create a square image. By default the 
# color of the padded pixels is black. These images are then saved to a new folder.
###

# Split char used in the CSV
SPLITCHAR = ","

# This function takes an image and a pixel size. It resizes the largest dimension to the desired size and then pads
# the remaining pixels with padColor to create a square image. It returns the square image.
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh = size
    sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

# Main Function: Takes the input command arguments and reads all the images listed in the image list.
# It resizeAndPad all the images and then save them to a new folder.
def main(args):
	SIZE = args.image_size
    imagedir = os.path.join(os.getcwd(), args.input_directory)

    if not os.path.isdir(imagedir):
        sys.exit("Image directory does not exist: {}".format(imagedir))

    imagelistfile = os.path.join(os.getcwd(), args.input_list_file)

    if not os.path.isfile(imagelistfile):
        sys.exit("Image list file does not exist: {}".format(imagelistfile))

    # Reads the annotation file for the list of images
    annotations = []
    with open(imagelistfile, 'r') as infile:
        annotations = infile.readlines()

    cwd = os.getcwd()
    outdir = os.path.join(cwd, arg_dict.output_directory)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Annotations are: imagefile
    for annot in annotations:
        imagepath = os.path.join(imagedir, annot.strip())

        image = cv2.imread(imagepath)
        if image is not None:
            resized = resizeAndPad(image, SIZE)
            
            label = annot[0]
            classdir = os.path.join(outdir, label)
            if not os.path.exists(classdir):
                os.mkdir(classdir)
                
            cv2.imwrite(os.path.join(outdir, annot.strip()),resized)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_list_file", default="scraped/imagelist.csv", help="name of input image list file")
    parser.add_argument("--input_directory", default="scraped", help="name of directory that contains images")
    parser.add_argument("--output_directory", default="resized2", help="name of input image list file")
    parser.add_argument("--image_size", default=64, type = int, help="size of input images")
    args = parser.parse_args()
    main(args)
