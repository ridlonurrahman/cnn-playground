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

def batch_norm(x, depth, training):
    """
    Batch normalisation
    Args:
        x:           inputs
        depth:       depth
        training:    training True or False
    Return:
        batch_normalized:      batch-normalized maps
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

def inference(images, NUM_CLASSES, training=True):
    """Build the CIFAR-10 model.
    Args:
      images: Images
      NUM_CLASSES = 10 or 100
      training = removed in this function

    Returns:
      Logits.
    """

    # CONV_LAYER 1
    with tf.variable_scope('conv1_1') as scope:
        kernel = tf_variable_l2('weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [64], tf.constant_initializer(0.0))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn1_1 = batch_norm(output_layer, 64, training)
        conv1_1 = tf.nn.relu(bn1_1, name=scope.name)

    with tf.variable_scope('conv1_2') as scope:
        kernel = tf_variable_l2('weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [64], tf.constant_initializer(0.1))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn1_2 = batch_norm(output_layer, 64, training)
        conv1_2 = tf.nn.relu(bn1_2, name=scope.name)

    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    if training==True:
      dropout_1 = tf.nn.dropout(pool1, 0.75)
    else:
      dropout_1 = tf.nn.dropout(pool1, 1)

    # CONV_LAYER 2
    with tf.variable_scope('conv2_1') as scope:
        kernel = tf_variable_l2('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(dropout_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [128], tf.constant_initializer(0.0))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn2_1 = batch_norm(output_layer, 128, training)
        conv2_1 = tf.nn.relu(bn2_1, name=scope.name)

    with tf.variable_scope('conv2_2') as scope:
        kernel = tf_variable_l2('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [128], tf.constant_initializer(0.1))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn2_2 = batch_norm(output_layer, 128, training)
        conv2_2 = tf.nn.relu(bn2_2, name=scope.name)

    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    if training==True:
      dropout_2 = tf.nn.dropout(pool2, 0.75)
    else:
      dropout_2 = tf.nn.dropout(pool2, 1)

    # CONV_LAYER 3
    with tf.variable_scope('conv3_1') as scope:
        kernel = tf_variable_l2('weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(dropout_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [256], tf.constant_initializer(0.0))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn3_1 = batch_norm(output_layer, 256, training)
        conv3_1 = tf.nn.relu(bn3_1, name=scope.name)

    with tf.variable_scope('conv3_2') as scope:
        kernel = tf_variable_l2('weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [256], tf.constant_initializer(0.1))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn3_2 = batch_norm(output_layer, 256, training)
        conv3_2 = tf.nn.relu(bn3_2, name=scope.name)

    with tf.variable_scope('conv3_3') as scope:
        kernel = tf_variable_l2('weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
        conv_layer = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf_variable('biases', [256], tf.constant_initializer(0.1))
        output_layer = tf.nn.bias_add(conv_layer, biases)
        bn3_3 = batch_norm(output_layer, 256, training)
        conv3_3 = tf.nn.relu(bn3_3, name=scope.name)


    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    if training==True:
      dropout_3 = tf.nn.dropout(pool3, 0.75)
    else:
      dropout_3 = tf.nn.dropout(pool3, 1)

    # fc_3
    with tf.variable_scope('fc_3') as scope:
        reshape = tf.reshape(dropout_3, [128, -1])
        dim = reshape.get_shape()[1].value
        weights = tf_variable_l2('weights', shape=[dim, 1024], stddev=0.05, wd=0.0005)
        biases = tf_variable('biases', [1024], tf.constant_initializer(0.1))
        output_layer = tf.matmul(reshape, weights) + biases
        bn4 = batch_norm(output_layer, 1, training)
        fc_3 = tf.nn.relu(bn4, name=scope.name)

    if training==True:
      dropout_4 = tf.nn.dropout(fc_3, 0.5)
    else:
      dropout_4 = tf.nn.dropout(fc_3, 1)

    # fc_4
    with tf.variable_scope('fc_4') as scope:
        weights = tf_variable_l2('weights', shape=[1024, 1024], stddev=0.05, wd=0.0005)
        biases = tf_variable('biases', [1024], tf.constant_initializer(0.1))
        output_layer = tf.matmul(dropout_4, weights) + biases
        bn5 = batch_norm(output_layer, 1, training)
        fc_4 = tf.nn.relu(bn5, name=scope.name)

    if training==True:
      dropout_5 = tf.nn.dropout(fc_4, 0.5)
    else:
      dropout_5 = tf.nn.dropout(fc_4, 1)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf_variable_l2('weights', [1024, NUM_CLASSES], stddev=1/1024.0, wd=0.0)
        biases = tf_variable('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(dropout_5, weights), biases, name=scope.name)

    return softmax_linear
