from bs4 import BeautifulSoup
import requests
import re
import os
import argparse
import sys
import json
from PIL import Image
from io import BytesIO
import urllib.request, random, imghdr, time

###
# adapted from http://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search
# This script takes in queries to search on google images. It then takes the first 20 images of the query and
# attempts to download it. The queries can be inputed through argparse or by editing the file directly.
###

# Send the request to the url, using the input header
def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url,headers=header)),'html.parser')

# Main function: Queries google images with each input query and downloads images
def main(args):
        default_parent_folder = "scraped_images"

        cwd = os.getcwd()
        parent_folder = os.path.join(cwd, default_parent_folder)
        if not os.path.exists(parent_folder):
            os.mkdir(parent_folder)
        if args.query is not None:
            class_name = args.folder_name
            queries = args.query
        else:
            class_name = "dog"
            queries = ["dogs", "pug", "puppy"]

        save_directory = os.path.join(parent_folder, class_name)

        # for each of the input query, perform search and save the image.
        for query in queries:
                max_images = 100
                query= query.replace(" ", "+") 
                url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
                header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
                soup = get_soup(url,header)
                ActualImages=[]# contains the link for Large original images, type of  image
                for a in soup.find_all("div",{"class":"rg_meta"}):
                    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
                    ActualImages.append((link,Type))
                counter = 0
                for i , (img , _) in enumerate( ActualImages):
                        try:
                                raw_img = Image.open(BytesIO(requests.get(img, timeout=30).content))
                                raw_img.save(os.path.join(save_directory, "{}{}.{}".format(query, counter, raw_img.format.lower())))
                                counter += 1
                                if counter >= max_images:
                                    break
                                raw_img.close()
                        except:
                                pass
                print("Cycle for " + query + " done")
        print("All Images have been downloaded.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", nargs='+', help="name of query to search for")
    parser.add_argument("--folder_name", default="query", help="name of folder to store scraped images")
    args = parser.parse_args()
    main(args)
    sys.exit()
