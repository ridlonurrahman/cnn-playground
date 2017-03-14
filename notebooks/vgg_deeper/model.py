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

def inference(images, NUM_CLASSES):
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
    conv1_1 = tf.nn.relu(pre_activation, name=scope.name)

  with tf.variable_scope('conv1_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(pre_activation, name=scope.name)

  pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # CONV 2
  with tf.variable_scope('conv2_1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 64, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(pre_activation, name=scope.name)

  with tf.variable_scope('conv2_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 128],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(pre_activation, name=scope.name)

  pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  # CONV 3
  with tf.variable_scope('conv3_1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 128, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(pre_activation, name=scope.name)

  with tf.variable_scope('conv3_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(pre_activation, name=scope.name)

  with tf.variable_scope('conv3_3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 256],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(pre_activation, name=scope.name)

  pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  # CONV 4
  with tf.variable_scope('conv4_1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 256, 512],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(pre_activation, name=scope.name)

  with tf.variable_scope('conv4_2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 512, 512],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(pre_activation, name=scope.name)

  with tf.variable_scope('conv4_3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 512, 512],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(pre_activation, name=scope.name)

  pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')

  # local3
  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool4, [128, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
                                          stddev=1/1024.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return softmax_linear