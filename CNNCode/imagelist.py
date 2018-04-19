import sys
import argparse
import os
import matplotlib.pyplot as plt
import random

# Split char used in the CSV
SPLITCHAR = ";"

def makecsv(subdir, files):
    csvfile = 'GT_{}.csv'.format(subdir)
    with open(subdir + "/" + csvfile, 'w') as outf:
        for fi in files:
            outf.write(fi+"\n")

    return csvfile

def main(arg_dict):

    # Output data
    imagelist = []
    imagecount = {}

    # Finding the Ground Truth csv files in each class folder
    cwd = os.getcwd()
    subdirs = [subdir for subdir in os.listdir(cwd) if os.path.isdir(subdir)]
    subdirs.sort()
    for subdir in subdirs:
        newcwd = os.path.join(cwd, subdir)
        files = os.listdir(newcwd)

        csvfiles = [csvfile for csvfile in files if csvfile.endswith(".csv")]
        print ("In Directory {}".format(subdir))

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

    # Writing files
    with open(arg_dict.output_list_file, 'w') as outfile:
        outfile.writelines(imagelist)

    with open(arg_dict.output_summary_file, 'w') as outfile:
        outfile.write("CLASSLABELS = {}\n".format(len(imagecount)))
        outfile.write("CLASS\tNUMIMAGES\n")
        for key in sorted(imagecount.keys()):
            outfile.write("{}\t{}\n".format(key,imagecount[key]))
    
    plt.bar([key + " " for key in imagecount.keys()], imagecount.values())
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
