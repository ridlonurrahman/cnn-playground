import tensorflow as tf
import numpy as np

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if 0 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if 0 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def inference(images, NUM_CLASSES, training=True):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    # CONV 1
    with tf.variable_scope('conv1_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        #bn1_1 = tf.contrib.layers.batch_norm(pre_activation, center=True, scale=True, is_training=training)
        conv1_1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('conv1_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        #bn1_2 = tf.contrib.layers.batch_norm(pre_activation, center=True, scale=True, is_training=training)
        conv1_2 = tf.nn.relu(pre_activation, name=scope.name)

    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    if training==True:
      dropout_1 = tf.nn.dropout(pool1, 0.75)
    else:
      dropout_1 = tf.nn.dropout(pool1, 1)

    # CONV 2
    with tf.variable_scope('conv2_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 128],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(dropout_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        #bn2_1 = tf.contrib.layers.batch_norm(pre_activation, center=True, scale=True, is_training=training)
        conv2_1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('conv2_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 128],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        #bn2_2 = tf.contrib.layers.batch_norm(pre_activation, center=True, scale=True, is_training=training)
        conv2_2 = tf.nn.relu(pre_activation, name=scope.name)

    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    if training==True:
      dropout_2 = tf.nn.dropout(pool2, 0.75)
    else:
      dropout_2 = tf.nn.dropout(pool2, 1)

    # CONV 3
    with tf.variable_scope('conv3_1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 128, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(dropout_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        #bn3_1 = tf.contrib.layers.batch_norm(pre_activation, center=True, scale=True, is_training=training)
        conv3_1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('conv3_2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        #bn3_2 = tf.contrib.layers.batch_norm(pre_activation, center=True, scale=True, is_training=training)
        conv3_2 = tf.nn.relu(pre_activation, name=scope.name)

    pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')

    if training==True:
      dropout_3 = tf.nn.dropout(pool3, 0.75)
    else:
      dropout_3 = tf.nn.dropout(pool3, 1)

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(dropout_3, [128, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 512],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.matmul(reshape, weights) + biases
        if training==True:
          bn4 = tf.contrib.layers.batch_norm(pre_activation, center=False, updates_collections=None, is_training=True)
        else:
          bn4 = tf.contrib.layers.batch_norm(pre_activation, center=False, updates_collections=None, is_training=False)
        local3 = tf.nn.relu(bn4, name=scope.name)

    if training==True:
      dropout_4 = tf.nn.dropout(local3, 0.5)
    else:
      dropout_4 = tf.nn.dropout(local3, 1)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[512, 512],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.matmul(dropout_4, weights) + biases
        if training==True:
          bn5 = tf.contrib.layers.batch_norm(pre_activation, center=False, updates_collections=None, is_training=True)
        else:
          bn5 = tf.contrib.layers.batch_norm(pre_activation, center=False, updates_collections=None, is_training=False)
        local4 = tf.nn.relu(bn5, name=scope.name)

    if training==True:
      dropout_5 = tf.nn.dropout(local4, 0.5)
    else:
      dropout_5 = tf.nn.dropout(local4, 1)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES],
                                              stddev=1/512.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(dropout_5, weights), biases, name=scope.name)

    return softmax_linear
