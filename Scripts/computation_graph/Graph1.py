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
    # The input to this layer is 32 x 32 x 3
    encoder_layer1 = tf.layers.conv2d(inputs, 8, [5, 5], strides=(2, 2), padding="SAME")
    # The output from this layer would be 16 x 16 x 8

    # The input to this layer is same as encoder_layer1 output: 16 x 16 x 8
    encoder_layer2 = tf.layers.conv2d(encoder_layer1, 16, [5, 5], strides=(2, 2), padding="SAME")
    # The output would be: 8 x 8 x 16

    # The input is same as above output: 8 x 8 x 16
    encoder_layer3 = tf.layers.conv2d(encoder_layer2, 32, [5, 5], strides=(4, 4), padding="SAME")
    # The output would be: 2 x 2 x 32
    # This is the latent representation of the input that is 128 dimensional.
    # Compression achieved from 32 x 32 x 3 i.e 3072 dimensions to 2 x 2 x 32 i. e. 128

    # decoder layers:
    # The input to this layer is 2 x 2 x 32
    decoder_layer1 = tf.layers.conv2d_transpose(encoder_layer3, 32, [5, 5], strides=(4, 4), padding="SAME")
    # Output from this layer: 8 x 8 x 32

    # The input to this layer: 8 x 8 x 32
    decoder_layer2 = tf.layers.conv2d_transpose(decoder_layer1, 16, [5, 5], strides=(2, 2), padding="SAME")
    # output from this layer: 16 x 16 x 16

    # The input of this layer: 16 x 16 x 16
    decoder_layer3 = tf.layers.conv2d_transpose(decoder_layer2, 3, [5, 5], strides=(2, 2), padding="SAME")
    # output of this layer: 32 x 32 x 3 # no. of channels are adjusted

    output = tf.identity(encoder_layer3, name = "encoded_representation") # the latent representation of the input image.


    y_pred = tf.identity(decoder_layer3, name = "prediction") # output of the decoder
    y_true = inputs # input at the beginning

    # define the loss for this model:
    # calculate the loss and optimize the network
    loss = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)), name="loss") # claculate the mean square error loss

    train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss, name="train_op") # using Adam optimizer for optimization
