# The tf_variable and tf_variable_l2 are available in Tensorflow's example code.
# We use that so we can train using multiple GPU
# Variable weight decay is to apply L2

import tensorflow as tf
import numpy as np

def convolutional_layer_bn(names, inputs, training, shapes, strides, paddings='SAME', bias_init=0.1):
    """ Convolutional layer
    Args:
        names: scope name
        inputs: input layer
        shapes: [h_kernel, w_kernel, d_input, d_output]
        strides: [1, h, w, 1]
        paddings: SAME or VALID
        bias_init: bias initialization

    Returns:
        convolutional layer
    """
    with tf.variable_scope(names) as scope:
        kernel = tf_variable_l2('weights', shape=shapes, stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(inputs, kernel, strides, padding=paddings)
        biases = tf_variable('biases', shapes[3], tf.constant_initializer(bias_init))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn = batch_norm(output_layer, shapes[3], training)
        conv_layer = tf.nn.elu(bn, name=scope.name)
    return conv_layer

def fully_connected_layer_bn(names, inputs, training, shapes, stddevs, wd, bias_init=0.1):
    """ Fully connected layer
    Args:
        names: scope name
        inputs: input layer
        shapes: [input, output]
        stddevs: stddev for weight init
        wd: weight decay
        bias_init: bias initialization

    Returns:
        fully connected layer
    """
    with tf.variable_scope(names) as scope:
        weights = tf_variable_l2('weights', shape=shapes, stddev=stddevs, wd=wd)
        biases = tf_variable('biases', shapes[1], tf.constant_initializer(bias_init))
        output_layer = tf.matmul(inputs, weights) + biases
        bn = batch_norm(output_layer, 1, training)
        fc_layer = tf.nn.elu(bn, name=scope.name)
    return fc_layer

def batch_norm(x, depth, training):
    """ Batch normalisation
    Args:
        x: inputs
        depth: depth
        training: training or evaluation
    Return:
        batch normalized
    """
    with tf.variable_scope('bn'):
        if depth == 1:
            gamma = tf.Variable(tf.ones([x.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))
            batch_mean, batch_var = tf.nn.moments(x,[0], name='moments')
        else:
            beta = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        if training==True:
            mean, var = mean_var_update()
        else:
            mean = batch_mean
            var = batch_var

        batch_normalized = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return batch_normalized

def dropout(inputs, keep_prob, training):
    """ Dropout
    Args:
        inputs: layers to dropout
        keep_prob: 1 - probability of dropped out units
        training: training or evaluation

    Returns:
        Dropout
    """
    if training==True:
      dropouts = tf.nn.dropout(inputs, keep_prob)
    else:
      dropouts = tf.nn.dropout(inputs, 1)
    return dropouts

def tf_variable(name, shape, initializer):
    """
    Args:
      name: name
      shape: shape of var
      initializer: initializer

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def tf_variable_l2(name, shape, stddev, wd):
    """
    Args:
      name: name
      shape: shape of var
      stddev: tddev of a truncated Gaussian
      wd: add L2Loss or not when none

    Returns:
      Variable Tensor
    """
    var = tf_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(images, NUM_CLASSES, training=True):
    """Build the CIFAR model.
    Args:
      images: Images
      NUM_CLASSES = 10 or 100
      training = removed in this function

    Returns:
      Logits.
    """

    # CONV_LAYER 64
    conv1_1 = convolutional_layer_bn('conv1_1', images, training, shapes=[3, 3, 3, 64], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.0)
    conv1_2 = convolutional_layer_bn('conv1_2', conv1_1, training, shapes=[3, 3, 64, 64], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.1)
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    dropout_1 = dropout(pool1, 0.75, training)

    # CONV_LAYER 128
    conv2_1 = convolutional_layer_bn('conv2_1', dropout_1, training, shapes=[3, 3, 64, 128], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.0)
    conv2_2 = convolutional_layer_bn('conv2_2', conv2_1, training, shapes=[3, 3, 128, 128], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.1)
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    dropout_2 = dropout(pool2, 0.75, training)

    # CONV_LAYER 256
    conv3_1 = convolutional_layer_bn('conv3_1', dropout_2, training, shapes=[3, 3, 128, 256], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.0)
    conv3_2 = convolutional_layer_bn('conv3_2', conv3_1, training, shapes=[3, 3, 256, 256], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.1)
    # We use 6 conv layers in the report
    conv3_3 = convolutional_layer_bn('conv3_3', conv3_2, training, shapes=[3, 3, 256, 256], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.1)
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    dropout_3 = dropout(pool3, 0.75, training)

    # FC
    reshape = tf.reshape(dropout_3, [128, -1])
    dim = reshape.get_shape()[1].value
    fc_3 = fully_connected_layer_bn('fc_3', reshape, training, shapes=[dim, 1024], stddevs=0.05, wd=0.0005, bias_init=0.1)
    dropout_4 = dropout(fc_3, 0.5, training)

    # FC
    fc_4 = fully_connected_layer_bn('fc_4', dropout_4, training, shapes=[1024, 1024], stddevs=0.05, wd=0.0005, bias_init=0.1)
    dropout_5 = dropout(fc_4, 0.5, training)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf_variable_l2('weights', [1024, NUM_CLASSES], stddev=1/1024.0, wd=0.0)
        biases = tf_variable('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(dropout_5, weights), biases, name=scope.name)

    return softmax_linear
