import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models_depth


def init_depth_map(sess):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models_depth.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    # Load the converted parameters
    print('Loading the model')

    # Use to load from ckpt file
    saver = tf.train.Saver()
    saver.restore(sess, "../../weights/NYU_FCRN.ckpt")

    return input_node, net



def predict_depth_map(image, sess, input_node, net):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    img = image.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # Prediction

    pred = None

    # Evalute the network for the given image
    pred = sess.run(net.get_output(), feed_dict={input_node: img})

    #sess.close()

    if False:
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        fig.savefig("depth.jpg")
    else:
        print("depth map computed")

    return pred






    '''
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models_depth.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    with tf.Graph().as_default() as net_graph:

        # Load the converted parameters
        print('Loading the model')

        sess = tf.Session(graph=net_graph)

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, "../../weights/NYU_FCRN.ckpt")

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})

        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()


    return pred
    '''
    #https://stackoverflow.com/questions/41607144/loading-two-models-from-saver-in-the-same-tensorflow-session
