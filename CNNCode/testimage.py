import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import cv2

INPUTOPERATION = "input"
OUTPUTOPERATION = "output"
IMAGESHAPE = 64
LABELS = ("SUV", "SEDAN", "TRUCK")
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def resizeAndPad(img, size, padColor=127):

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

def main(args):
    graph = load_graph(args.model_file)

    print ("Graph Loaded")

    input_name = "import/" + INPUTOPERATION
    output_name = "import/" + OUTPUTOPERATION
    input_operate = graph.get_operation_by_name(input_name)
    output_operate = graph.get_operation_by_name(output_name)

    cwd = os.getcwd()
    test_dir = os.path.join(cwd, args.test_data)
    files = os.listdir(test_dir)

    with tf.Session(graph = graph) as sess:
        for path in files:
            imagepath = os.path.join(test_dir, path) 
            image = cv2.imread(imagepath)
            resized = resizeAndPad(image, IMAGESHAPE)
            x_in = None

            if args.gray:
                x = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                x = x.astype('float32')
                x_in = (x - 128.) / 256.
                x_in = np.reshape(x, (-1, IMAGESHAPE, IMAGESHAPE, 1))
            else:
                x = resized[:, :, ::-1]
                x = x.astype('float32')
                x_in = (x - 128.) / 256.
                x_in = np.reshape(x, (-1, IMAGESHAPE, IMAGESHAPE, 3))

            results = sess.run(output_operate.outputs[0], {input_operate.outputs[0]: x_in})
            results = np.squeeze(results)
            print(LABELS[np.argmax(results)])
            cv2.imshow('frame', image)
            cv2.waitKey(-1)

    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="model.pb", help="name of file containing trained model")
    parser.add_argument("--test_data", default="testimages2", help="name of folder containing test images")
    parser.add_argument("--gray", action="store_true", help="gray scale model")
    args = parser.parse_args()
    main(args)
