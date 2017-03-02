import tensorflow as tf

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    '''Create a fully connected layer with ReLu as the activation function'''
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

def fully_connected_layer_sigmoid(inputs, input_dim, output_dim, nonlinearity=tf.sigmoid):
    '''Create a fully connected layer with Sigmoid as the activation function'''
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

def fully_connected_layer_tanh(inputs, input_dim, output_dim, nonlinearity=tf.tanh):
    '''Create a fully connected layer with Tanh as the activation function'''
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

def layer_with_batch_norm(inputs, input_dim, output_dim, num_hidden, phase_train, nonlinearity=tf.nn.relu):
    '''Create a fully connected layer with batch normalisation'''
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')

    if phase_train==0:
        outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    else:
        u = tf.matmul(inputs, weights) + biases
        means, variances = tf.nn.moments(u, [0])
        u_hat = (u-means) / tf.sqrt(variances+10**-8)
        gamma = tf.Variable(tf.ones([num_hidden]))
        beta = tf.Variable(tf.zeros([num_hidden]))
        bn = u_hat * gamma + beta
        outputs = nonlinearity(bn)
    return outputs

def batchnorm_dropout(inputs, input_dim, output_dim, num_hidden, phase_train, nonlinearity=tf.nn.relu):
    '''Create a fully connected layer with batch normalisation and dropout'''
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')

    if phase_train==0:
        outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
        #outputs = tf.nn.dropout(outputs_activation, 0.6)
    else:
        u = tf.matmul(inputs, weights) + biases
        means, variances = tf.nn.moments(u, [0])
        u_hat = (u-means) / tf.sqrt(variances+10**-8)
        gamma = tf.Variable(tf.ones([num_hidden]))
        beta = tf.Variable(tf.zeros([num_hidden]))
        bn = u_hat * gamma + beta
        outputs_activation = nonlinearity(bn)
        outputs = tf.nn.dropout(outputs_activation, 0.8)
    return outputs



