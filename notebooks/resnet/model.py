import tensorflow as tf
import numpy as np
import six

# TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
def _batch_norm(name, x, training=True):
  """Batch normalization."""
  with tf.variable_scope(name):
    params_shape = [x.get_shape()[-1]]

    beta = tf.get_variable(
        'beta', params_shape, tf.float32,
        initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(
        'gamma', params_shape, tf.float32,
        initializer=tf.constant_initializer(1.0, tf.float32))

    if training==True:
      mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

      moving_mean = tf.get_variable(
          'moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32),
          trainable=False)
      moving_variance = tf.get_variable(
          'moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32),
          trainable=False)

      '''_extra_train_ops.append(moving_averages.assign_moving_average(
                            moving_mean, mean, 0.9))
                        _extra_train_ops.append(moving_averages.assign_moving_average(
                            moving_variance, variance, 0.9))'''
    else:
      mean = tf.get_variable(
          'moving_mean', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32),
          trainable=False)
      variance = tf.get_variable(
          'moving_variance', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32),
          trainable=False)
    # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
    y = tf.nn.batch_normalization(
        x, mean, variance, beta, gamma, 0.001)
    y.set_shape(x.get_shape())
    return y

def _residual(x, in_filter, out_filter, stride,
              activate_before_residual, training):
  """Residual unit with 2 sub layers."""
  with tf.variable_scope('residual_only_activation'):
    orig_x = x
    x = _batch_norm('init_bn', x, training)
    x = _relu(x, 0.1)

  with tf.variable_scope('sub1'):
    x = _conv('conv1', x, 3, in_filter, out_filter, stride)

  with tf.variable_scope('sub2'):
    x = _batch_norm('bn2', x, training)
    x = _relu(x, 0.1)
    x = _conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

  with tf.variable_scope('sub_add'):
    if in_filter != out_filter:
      orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
      orig_x = tf.pad(
          orig_x, [[0, 0], [0, 0], [0, 0],
                   [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
    x += orig_x

  tf.logging.debug('image after unit %s', x.get_shape())
  return x

def _decay(self):
  """L2 weight decay loss."""
  costs = []
  for var in tf.trainable_variables():
    if var.op.name.find(r'DW') > 0:
      costs.append(tf.nn.l2_loss(var))
      # tf.summary.histogram(var.op.name, var)

  return tf.multiply(0.0002, tf.add_n(costs))

def _stride_arr(stride):
  """Map a stride scalar to the stride array for tf.nn.conv2d."""
  return [1, stride, stride, 1]

def _conv(name, x, filter_size, in_filters, out_filters, strides):
  """Convolution."""
  with tf.variable_scope(name):
    n = filter_size * filter_size * out_filters
    kernel = tf.get_variable(
        'DW', [filter_size, filter_size, in_filters, out_filters],
        tf.float32, initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0/n)))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME')

def _relu(x, leakiness=0.0):
  """Relu, with optional leaky support."""
  return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def _fully_connected(x, out_dim):
  """FullyConnected layer for final output."""
  x = tf.reshape(x, [128, -1])
  w = tf.get_variable(
      'DW', [x.get_shape()[1], out_dim],
      initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
  b = tf.get_variable('biases', [out_dim],
                      initializer=tf.constant_initializer())
  return tf.nn.xw_plus_b(x, w, b)

def _global_avg_pool(x):
  assert x.get_shape().ndims == 4
  return tf.reduce_mean(x, [1, 2])

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

  with tf.variable_scope('init'):
    x = images
    x = _conv('init_conv', x, 3, 3, 16, _stride_arr(1))

  num_res_units = 5
  strides = [1, 2, 2]
  activate_before_residual = [True, False, False]
  filters = [16, 160, 320, 640]
    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    # filters = [16, 160, 320, 640]
    # Update hps.num_residual_units to 9

  with tf.variable_scope('unit_1_0'):
    x = _residual(x, filters[0], filters[1], _stride_arr(strides[0]),
                 activate_before_residual[0], training)
  for i in six.moves.range(1, num_res_units):
    with tf.variable_scope('unit_1_%d' % i):
      x = _residual(x, filters[1], filters[1], _stride_arr(1), False, training)

  with tf.variable_scope('unit_2_0'):
    x = _residual(x, filters[1], filters[2], _stride_arr(strides[1]),
                 activate_before_residual[1], training)
  for i in six.moves.range(1, num_res_units):
    with tf.variable_scope('unit_2_%d' % i):
      x = _residual(x, filters[2], filters[2], _stride_arr(1), False, training)

  with tf.variable_scope('unit_3_0'):
    x = _residual(x, filters[2], filters[3], _stride_arr(strides[2]),
                 activate_before_residual[2], training)
  for i in six.moves.range(1, num_res_units):
    with tf.variable_scope('unit_3_%d' % i):
      x = _residual(x, filters[3], filters[3], _stride_arr(1), False, training)

  with tf.variable_scope('unit_last'):
    x = _batch_norm('final_bn', x, training)
    x = _relu(x, 0.1)
    x = _global_avg_pool(x)

  with tf.variable_scope('logit'):
    softmax_linear = _fully_connected(x, NUM_CLASSES)

  return softmax_linear