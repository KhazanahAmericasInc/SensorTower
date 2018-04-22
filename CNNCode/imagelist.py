import sys
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random

### 
# This script is used to list all the image files in the directory
# It provides the functionality to shuffle the list and have a maximum sample size
# This is useful to feed into preprocessing as it summarizes which images will be used
###

# This function takes a directory and the list of files inside.
# It creates a csv file inside the directory that lists all the files
def makecsv(subdir, files):
    csvfile = 'GT_{}.csv'.format(subdir)
    with open(subdir + "/" + csvfile, 'w') as outf:
        for fi in files:
            outf.write(fi+"\n")

    return csvfile

# This is the main function. It takes the parsed dictionary from command arguments
# It iterates through all subdirectories in the folder to create final summary files
def main(arg_dict):

    # Output data
    imagelist = []
    imagecount = {}

    # Finding the Ground Truth csv files in each class folder
    cwd = os.getcwd()
    subdirs = [subdir for subdir in os.listdir(cwd) if os.path.isdir(subdir)]
    subdirs.sort()

    # Iterates through all subdirectories found
    for subdir in subdirs:
        newcwd = os.path.join(cwd, subdir)
        files = os.listdir(newcwd)

        csvfiles = [csvfile for csvfile in files if csvfile.endswith(".csv")]
        print ("In Directory {}".format(subdir))

        # If there are no csv files in the directory, create a new one.
        if len(csvfiles) == 0:
            print ("NOTE: Making csv file for directory {}".format(subdir))
            csvfiles = [makecsv(subdir, files)]
        elif len(csvfiles) > 1:
            print("ERROR: Directory {} has more than one csv file!".format(subdir))

        # Parsing csv file(s)
        for csvfile in csvfiles:
            filecontent = None
            with open(os.path.join(newcwd, csvfile), 'r') as infile:
                filecontent = infile.readlines()
            
            if arg_dict.shuffle:
                filecontent = random.sample(filecontent, len(filecontent))

            if arg_dict.maxsample != -1:
                try:
                    filecontent = filecontent[:arg_dict.maxsample]
                except:
                    print("Error setting max sample")
                    raise
            for line in filecontent:
                imagelist.append("{}/{}\n".format(subdir, line.strip()))
                if subdir in imagecount.keys():
                    imagecount[subdir] += 1
                else:
                    imagecount[subdir] = 1

    # Writing to summary list file
    with open(arg_dict.output_list_file, 'w') as outfile:
        outfile.writelines(imagelist)

    # Writing to the data summary file
    with open(arg_dict.output_summary_file, 'w') as outfile:
        outfile.write("CLASSLABELS = {}\n".format(len(imagecount)))
        outfile.write("CLASS\tNUMIMAGES\n")
        for key in sorted(imagecount.keys()):
            outfile.write("{}\t{}\n".format(key,imagecount[key]))

    # Creating the bar graph
    x = np.arange(len(imagecount.keys()))
    plt.bar(x, imagecount.values())
    plt.xticks(x, imagecount.keys())
    plt.savefig(arg_dict.output_class_hist)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_list_file", default="imagelist.csv", help="name of output image list file")
    parser.add_argument("--output_summary_file", default="datasummary.txt", help="name of output summary file")
    parser.add_argument("--output_class_hist", default="classhistogram.png", help="name of output class histogram image")
    parser.add_argument("--maxsample", default=-1, type=int, help="specify maximum number of images per class")
    parser.add_argument("--shuffle", action="store_true", help="option to shuffle data")
    args = parser.parse_args()
    main(args)
