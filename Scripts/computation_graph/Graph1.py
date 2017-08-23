'''
    First graph model to be trained for this task.
    This file defines the method required to spawn and return a tensorflow graph for the autoencoder model.

    coded by: Animesh
'''

import tensorflow as tf


graph = tf.Graph() #create a new graph object

with graph.as_default():
    # define the computations of this graph here:

    # placeholder for the input data batch
    inputs = tf.placeholder(dtype= tf.float32, shape=(None, 32, 32, 3), name="inputs")


    # encoder layers:
    encoder_layer1 = tf.layers.conv2d(inputs, 8, [5, 5], strides=(2, 1), padding="SAME")
    encoder_layer2 = tf.layers.conv2d(encoder_layer1, 16, [5, 5], strides=(2, 1), padding="SAME")
    encoder_layer3 = tf.layers.conv2d(encoder_layer2, 32, [5, 5], strides=(4, 1), padding="SAME")

    # decoder layers:
    decoder_layer1 = tf.layers.conv2d_transpose(encoder_layer3, 32, [5, 5], strides=(4, 1), padding="SAME")
    decoder_layer2 = tf.layers.conv2d_transpose(decoder_layer1, 16, [5, 5], strides=(2, 1), padding="SAME")
    decoder_layer3 = tf.layers.conv2d_transpose(decoder_layer2, 1, [5, 5], strides=(2, 1), padding="SAME", activation=tf.nn.tanh)

    output = encoder_layer3 # the latent representation of the input image.


    y_pred = tf.identity(decoder_layer3, name = "prediction") # output of the decoder
    y_true = inputs # input at the beginning

    # define the loss for this model:
    # calculate the loss and optimize the network
    loss = tf.div(tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true))), 2, name="loss")  # claculate the mean square error loss

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, name="train_op") # using Adam optimizer for optimization
