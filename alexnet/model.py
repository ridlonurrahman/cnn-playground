# The tf_variable and tf_variable_l2 are available in Tensorflow's example code.
# We use that so we can train using multiple GPU
# Variable weight decay is to apply L2

import tensorflow as tf
import numpy as np

def convolutional_layer(names, inputs, shapes, strides, paddings='SAME', bias_init=0.1):
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
        conv_layer = tf.nn.relu(output_layer, name=scope.name)
    return conv_layer

def fully_connected_layer(names, inputs, shapes, stddevs, wd, bias_init=0.1):
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
        fc_layer = tf.nn.relu(tf.matmul(inputs, weights) + biases, name=scope.name)
    return fc_layer

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

def inference(images, NUM_CLASSES):
    """Build the CIFAR model.
    Args:
      images: Images
      NUM_CLASSES = 10 or 100
      training = removed in this function

    Returns:
      Logits.
    """

    # conv_layer1
    conv_layer1 = convolutional_layer('conv_layer1', images, shapes=[5, 5, 3, 24], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.1)
    # lrn1
    lrn1 = tf.nn.lrn(conv_layer1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    # pool1
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv_layer2
    conv_layer2 = convolutional_layer('conv_layer2', pool1, shapes=[5, 5, 24, 64], strides=[1, 1, 1, 1], paddings='SAME', bias_init=0.1)
    # lrn2
    lrn2 = tf.nn.lrn(conv_layer2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    # pool2
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # fc_1
    reshape = tf.reshape(pool2, [128, -1])
    dim = reshape.get_shape()[1].value
    fc_1 = fully_connected_layer('fc_1', reshape, shapes=[dim, 256], stddevs=1/256.0, wd=0.0005, bias_init=0.1)

    # fc_2
    fc_2 = fully_connected_layer('fc_2', fc_1, shapes=[256, 256], stddevs=1/256.0, wd=0.0005, bias_init=0.1)

    # softmax_linear       
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf_variable_l2('weights', [256, NUM_CLASSES], stddev=1/128.0, wd=0.0)
        biases = tf_variable('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc_2, weights), biases, name=scope.name)

    return softmax_linear
