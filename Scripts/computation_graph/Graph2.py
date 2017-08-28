'''
    Second graph model to be trained for this task.
    This file defines the method required to spawn and return a tensorflow graph for the autoencoder model.

    New Features: Use of a pooling layer and the corresponding unpooling layer.

    coded by: Animesh
'''

import tensorflow as tf


graph = tf.Graph() #create a new graph object

with graph.as_default():
    # define the computations of this graph here:

    ''' Helper functions for the graph building: '''
    def unpool(value, name='unpool'):
        """N-dimensional version of the unpooling operation from
        https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

        This function has been taken from link-> https://github.com/tensorflow/tensorflow/issues/2169

        :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
        :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
        """
        with tf.name_scope(name) as scope:
            sh = value.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(value, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat([out, tf.zeros_like(out)], i)
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name=scope)
        return out


    # placeholder for the input data batch
    inputs = tf.placeholder(dtype= tf.float32, shape=(None, 32, 32, 3), name="inputs")


    # encoder layers:
    # The input to this layer is 32 x 32 x 3
    encoder_layer1 = tf.layers.conv2d(inputs, 8, [5, 5], strides=(1, 1), padding="SAME")
    # The output from this layer would be 32 x 32 x 8 # no reduction in the dimensions

    # This pool layer reduces the dimensions by half
    pool_encoder_layer1 = tf.layers.max_pooling2d(encoder_layer1, pool_size=(2, 2), strides=(2, 2), padding="SAME")
    # This will reduce the dimensions to 16 x 16 x 8

    # The input to this layer is same as encoder_layer1 output: 16 x 16 x 8
    encoder_layer2 = tf.layers.conv2d(pool_encoder_layer1, 16, [5, 5], strides=(1, 1), padding="SAME")
    # no reduction in dimensions. the output will be 16 x 16 x 16

    # This pool layer will again reduce the dimensions by half
    pool_encoder_layer2 = tf.layers.max_pooling2d(encoder_layer2, pool_size=(2, 2), strides=(2, 2), padding="SAME")
    # output dimensions: 8 x 8 x 16

    # The input is same as above output: 8 x 8 x 16
    encoder_layer3 = tf.layers.conv2d(pool_encoder_layer2, 32, [5, 5], strides=(4, 4), padding="SAME")
    # The output would be: 2 x 2 x 32
    # This is the latent representation of the input that is 128 dimensional.
    # Compression achieved from 32 x 32 x 3 i.e 3072 dimensions to 2 x 2 x 32 i. e. 128

    # decoder layers:
    # The input to this layer is 2 x 2 x 32
    decoder_layer1 = tf.layers.conv2d_transpose(encoder_layer3, 32, [5, 5], strides=(4, 4), padding="SAME")
    # Output from this layer: 8 x 8 x 32

    # use the unpool layer:
    unpool_decoder_layer2 = unpool(decoder_layer1)
    # The dimesions will get doubled. so o/p will be 16 x 16 x 32

    # The input to this layer: 16 x 16 x 32
    decoder_layer2 = tf.layers.conv2d_transpose(unpool_decoder_layer2, 16, [5, 5], strides=(1, 1), padding="SAME")
    # output from this layer: 16 x 16 x 16

    # This will again double the dimesions
    unpool_decoder_layer3 = unpool(decoder_layer2)
    # output from this layer: 32 x 32 x 16

    # The input of this layer: 32 x 32 x 16
    decoder_layer3 = tf.layers.conv2d_transpose(unpool_decoder_layer3, 3, [5, 5], strides=(1, 1), padding="SAME")
    # output of this layer: 32 x 32 x 3 # no. of channels are adjusted

    output = tf.identity(encoder_layer3, name = "encoded_representation") # the latent representation of the input image.


    y_pred = tf.identity(decoder_layer3, name = "prediction") # output of the decoder
    y_true = inputs # input at the beginning

    # define the loss for this model:
    # calculate the loss and optimize the network
    loss = tf.reduce_mean(tf.abs(y_pred - y_true), name="loss") # claculate the mean square error loss

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, name="train_op") # using Adam optimizer for optimization
