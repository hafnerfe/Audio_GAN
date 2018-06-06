"""
Defines some useful operations and layers. (For now just layers)


Created by Felix Hafner, last edited on June 5th, 2018.
Praktikum for the embedded intelligence for healtcare and wellbeing chair.
University of Augsburg (UNIA).
"""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("stddev", 0.02, '')

def batch_norm(is_train, tensor, scope_name):
    """
    Batch normalization layer
    :param is_train: Bool value. For this purpose: always true
    :param tensor: the input layer
    :param scope_name: ..
    :return: the output layer
    """

    with tf.variable_scope(scope_name):
        # Get the shape of mean, variance, beta, gamma
        mask_shape = [1] * len(tensor.get_shape())
        mask_shape[-1] = tensor.get_shape()[-1].value

        # Create trainable variables to hold beta and gamma
        beta = tf.get_variable('beta', mask_shape, initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', mask_shape, initializer=tf.constant_initializer(1.0))

        # Calculate the moments based on the individual batch.
        n_dims = len(tensor.get_shape())
        mean, variance = tf.nn.moments(x=tensor, axes=[i for i in range(0, n_dims - 1)], keep_dims=True)
        return tf.nn.batch_normalization(tensor, mean, variance, beta, gamma, variance_epsilon=0.001)


def conv_with_relu(inputs, filter_shape, name, max_pool=True):
    """
    Convolutional layer with relu activation function. Using max pooling layer at the end if max_pool=True
    :param inputs: The input layer
    :param filter_shape:
    :param name:
    :param max_pool:
    :return: the output layer
    """

    w = tf.get_variable(name=name + 'w', shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(stddev=FLAGS.stddev))
    b = tf.get_variable(name=name + 'b', shape=[filter_shape[3]], initializer=tf.constant_initializer(0.0))
    out = tf.nn.conv2d(inputs, w, strides=[1, 2, 2, 1], padding='SAME')
    out = batch_norm(True, out, name+'bn')
    out = tf.nn.relu(out+b)

    if max_pool:
        out = tf.nn.max_pool(out, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return out


def deconv(inputs, filter_shape, name):
    """
    Deconvolutional layer with strides [1,2,2,1] -> both axis get multiplied by two
    :param inputs: the input layer
    :param filter_shape:
    :param name:
    :return: the output layer
    """

    w = tf.get_variable(name=name + 'w', shape=filter_shape,
                        initializer=tf.random_normal_initializer(stddev=FLAGS.stddev))
    b = tf.get_variable(name=name + 'b', shape=filter_shape[2], initializer=tf.constant_initializer(0.0))
    out = tf.nn.conv2d_transpose(inputs, w, strides=[1, 2, 2, 1],
                                 output_shape=[FLAGS.batch_size, int(inputs.shape[1] * 2),
                                               int(inputs.shape[2] * 2), int(filter_shape[2])])
    return out + b


def fc(tensor, outdim, name):
    """
    fully connected layer
    """
    indim = tensor.shape[1].value
    w = tf.get_variable(name + 'w', [indim, outdim],tf.float32,
                        initializer=tf.random_normal_initializer(stddev=FLAGS.stddev))
    b = tf.get_variable(name + 'b', [1, outdim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(tensor, w) + b

