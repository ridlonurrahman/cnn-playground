# The tf_variable and tf_variable_l2 are available in Tensorflow's example code.
# We use that so we can train using multiple GPU
# Variable weight decay is to apply L2

import tensorflow as tf
import numpy as np

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
    """Build the CIFAR-10 model.
    Args:
      images: Images
      NUM_CLASSES = 10 or 100
      training = removed in this function

    Returns:
      Logits.
    """

    # conv_layer1
    with tf.variable_scope('conv_layer1') as scope:
        kernel = tf_variable_l2('weights',
                                             shape=[5, 5, 3, 24],
                                             stddev=5e-2,
                                             wd=0.0)
        conv_layer = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [24], tf.constant_initializer(0.1))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        conv_layer1 = tf.nn.relu(output_layer, name=scope.name)

    # lrn1
    lrn1 = tf.nn.lrn(conv_layer1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    # pool1
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv_layer2
    with tf.variable_scope('conv_layer2') as scope:
        kernel = tf_variable_l2('weights',
                                             shape=[5, 5, 24, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv_layer = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [64], tf.constant_initializer(0.1))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        conv_layer2 = tf.nn.relu(output_layer, name=scope.name)

    # lrn2
    lrn2 = tf.nn.lrn(conv_layer2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    # pool2
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # fc_1
    with tf.variable_scope('fc_1') as scope:
        reshape = tf.reshape(pool2, [128, -1])
        dim = reshape.get_shape()[1].value
        weights = tf_variable_l2('weights', shape=[dim, 256], stddev=1/256.0, wd=0.0005)
        biases = tf_variable('biases', [256], tf.constant_initializer(0.1))
        fc_1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # fc_2
    with tf.variable_scope('fc_2') as scope:
        weights = tf_variable_l2('weights', shape=[256, 256], stddev=1/128.0, wd=0.0005)
        biases = tf_variable('biases', [128], tf.constant_initializer(0.1))
        fc_2 = tf.nn.relu(tf.matmul(fc_1, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf_variable_l2('weights', [256, NUM_CLASSES], stddev=1/128.0, wd=0.0)
        biases = tf_variable('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc_2, weights), biases, name=scope.name)

    return softmax_linear
