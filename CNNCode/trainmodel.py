import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
import numpy as np
import pickle
import os
import math
import time
import argparse

###
# This script takes in a binary file that contains all the images and their respective labels and trains 
# a convolutional neural that classifies the images. It outputs the model into the desired directory. In
# the directory, there would be multiple files that represent the tensorflow checkpoint and a .pb file 
# that stores the model weights and structure. It is also possible to reload a checkpoint to resume or
# further train the network.
###

# global constants for the script
VALIDATIONSIZE = 900
BATCHSIZE = 100
LEARNINGRATE = 0.001
DECAY = 0.9
NUMEPOCH = 200

### Initializes placeholders (containing image data and their labels)
def input_placeholders(img_shape, num_labels):
    images_pl = tf.placeholder(tf.float32, shape=(None, img_shape[0], img_shape[1], img_shape[2]), name='input')
    labels_pl = tf.placeholder(tf.int32, shape=(None, num_labels))
    return images_pl, labels_pl

### Main network function : builds the graph 
def inference(x, keep_prob, num_labels):

    # Input is 64x64x3 
    net = slim.conv2d(x, 32, [3, 3], stride=2, activation_fn=tf.nn.relu6, scope='conv1')      
    net = slim.conv2d(net, 32, [3, 3], stride=1, activation_fn=tf.nn.relu6, scope='conv2')

    # 32x32x32
    net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu6, scope='conv3')      
    net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu6, stride=2,  scope='conv4')

    # 16x16x64
    net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1, activation_fn=tf.nn.relu6, scope="conv5_depth")
    net = slim.conv2d(net, 64, [1, 1], activation_fn=tf.nn.relu6, scope='conv5_pointwise')
    net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=2, activation_fn=tf.nn.relu6, scope='conv6_depth')
    net = slim.conv2d(net, 128, [1, 1], activation_fn=tf.nn.relu6, scope='conv6_pointwise')
    
    # 8x8x128
    net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1, activation_fn=tf.nn.relu6, scope="conv7_depth")
    net = slim.conv2d(net, 128, [1, 1], activation_fn=tf.nn.relu6, scope='conv7_pointwise')
    net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=2, activation_fn=tf.nn.relu6, scope='conv8_depth')
    net = slim.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu6, scope='conv8_pointwise')

    # 4x4x256
    net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=1, activation_fn=tf.nn.relu6, scope="conv9_depth")
    net = slim.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu6, scope='conv9_pointwise')
    net = slim.separable_conv2d(net, None, [3, 3], depth_multiplier=1, stride=2, activation_fn=tf.nn.relu6, scope='conv10_depth')
    net = slim.conv2d(net, 512, [1, 1], activation_fn=tf.nn.relu6, scope='conv10_pointwise')

    # 2x2x256
    net = slim.max_pool2d(net, [2, 2], scope='Maxpool') 
    
    # 1x1x256
    net = tf.nn.dropout(net, keep_prob)
    net = slim.conv2d(net, num_labels, [1, 1], activation_fn=None, scope="Conv2D_final")

    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

    return net

### Evaluates correct prediction accuracy
def evaluate(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

### Training image batch generation
def batch_gen(images, labels, batchsize):
    start_id = 0
    while start_id < images.shape[0]:
        image_out = images[start_id : start_id + batchsize]
        label_out = labels[start_id : start_id + batchsize]

        start_id += batchsize

        yield (np.array(image_out), np.array(label_out))

# Main function. Sets up the tensorflow graph and variables. Trains the model with user given data
def main(args):

	# Name of the output save files.
    OUTSAVE = args.output_directory + "/" + args.output_model_name + ".ckpt"
    OUTMODEL = args.output_directory + "/" + args.output_model_name + ".pb"

    ##### Preparing training data
    # Load training data
    train = pickle.load(open(args.train_data, 'rb'))
    x = train['features']
    y_ = train['label']

    # Normalize pixel intensities between -1 and 1
    x = x.astype('float32')
    x = (x - 128.) / 128.

    # One hot class labels
    num_labels = np.unique(y_).shape[0]
    y_one_hot = np.zeros((y_.shape[0], num_labels))
    for i, onehot in enumerate(y_one_hot):
        onehot[y_[i]] = 1.
    #####

    ##### Preparing test data
    # Load testing data
    if args.test_data is not None:
        testdata = pickle.load(open(args.test_data, 'rb'))
        x_test = testdata['features']
        y_test = testdata['label']

        # Normalize pixel intensities between -1 and 1
        x_test = x_test.astype('float32')
        x_test = (x_test - 128.) / 128.

        # One hot class labels
        y_one_hot_test = np.zeros((y_test.shape[0], num_labels))
        for i, onehot in enumerate(y_one_hot_test):
            onehot[y_test[i]] = 1.
    #####

    ##### Split between training and validation
    x_v = x[:VALIDATIONSIZE, ...]
    y_v = y_one_hot[:VALIDATIONSIZE]
    x_t = x[VALIDATIONSIZE:, ...]
    y_t = y_one_hot[VALIDATIONSIZE:]
    #####

    num_training_imgs = x.shape[0]
    image_shape = x.shape[1:]

    ##### Build Network
    with tf.Graph().as_default():
        # Initializing placeholders and variables
        images_pl, labels_pl = input_placeholders(image_shape, num_labels)
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        global_step = tf.Variable(0, trainable=False)

        # Building network graph
        prediction = inference(images_pl, keep_prob, num_labels)
        output = tf.nn.softmax(prediction, name="output")

        # Defining loss function to minimize 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_pl, logits=prediction))
        
        # Defining learning rate and optimizer
        lr = LEARNINGRATE
        if args.learning_rate == None:
            lr = args.learning_rate
        learning_rate = tf.train.exponential_decay(lr, global_step, 500, DECAY, staircase = True) 
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)

        # Defining accuracy calculation 
        accuracy = evaluate(prediction, labels_pl)

        # Initializing session
        sess = tf.Session()
        if args.previous_model is not None:
            global_vars = tf.global_variables() 
            Adams = [v for v in global_vars if "Adam" in v.name]
            Betas = [v for v in global_vars if "beta" in v.name]
            no_restore = Adams + Betas
            no_restore.append(global_step)
            load_saver = tf.train.Saver([v for v in global_vars if v not in no_restore])
            load_saver.restore(sess, args.previous_model)
            sess.run(tf.variables_initializer(no_restore))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        # Creating saver
        saver = tf.train.Saver()
        stepcount = 0
        start_time = time.time()

        ##### Running Training
        for epoch in range(NUMEPOCH):

            # Initialize batch image generator
            train_gen = batch_gen(x_t, y_t, BATCHSIZE)
            num_batches_per_epoch = x_t.shape[0] // BATCHSIZE

            for _ in range(num_batches_per_epoch):

                # Feed batched training images to train
                images_batch, labels_batch = next(train_gen)
                feed_dict = { images_pl : images_batch, labels_pl : labels_batch, keep_prob : 0.5}
                _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)

                stepcount += 1

                # Display training status
                if stepcount % 100 == 0:
                    duration = time.time() - start_time
                    print("Step %d: loss = %.4f. Time Elapsed = %.3f sec" % (stepcount, loss_value, duration))

                # Once every 1000 steps, display the validation results
                if stepcount % 1000 == 0:
                    feed_dict = { images_pl : x_v, labels_pl : y_v, keep_prob : 1.0}
                    validation_accuracy = sess.run(accuracy, feed_dict = feed_dict)
                    print("On Validation Dataset: Accuracy = %0.04f" % validation_accuracy)
                    if args.test_data is not None:
                        feed_dict = { images_pl : x_test, labels_pl: y_one_hot_test, keep_prob: 1.0}
                        testing_accuracy = sess.run(accuracy, feed_dict = feed_dict)
                        print("On Testing Dataset: Accuracy = %0.04f" % testing_accuracy)
                    
        print ("DONE TRAINING")
        #####

        # Save results to ckpt
        saver.save(sess, OUTSAVE)

        # Save network to a frozen model
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['output'])
        with tf.gfile.FastGFile(OUTMODEL, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    #####


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="traindata.p", help="name of file containing training images")
    parser.add_argument("--test_data", help="name of file containing test images")
    parser.add_argument("--output_model_name", default="my_model", help="name of output model")
    parser.add_argument("--output_directory", default="output", help="name of output directory")
    parser.add_argument("--previous_model", help="name of previous trained model to continue training")
    parser.add_argument("--learning_rate", help="learning rate")
    args = parser.parse_args()
    main(args)

