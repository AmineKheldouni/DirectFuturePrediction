import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models_depth

def predict_depth_map(image):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    img = image.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    sess2 = tf.Session()

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models_depth.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    #with tf.Session() as sess2:

    # Load the converted parameters
    print('Loading the model')

    # Use to load from ckpt file
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess2, "NYU_FCRN.ckpt")

    # Use to load from npy file
    # net.load(model_data_path, sess)

    # Evalute the network for the given image
    pred = sess2.run(net.get_output(), feed_dict={input_node: img})

    # Plot result
    fig = plt.figure()
    ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
    fig.colorbar(ii)
    plt.show()

    sess2.close()

    return pred

    #https://stackoverflow.com/questions/41607144/loading-two-models-from-saver-in-the-same-tensorflow-session