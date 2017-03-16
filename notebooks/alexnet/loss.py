import tensorflow as tf


def loss(logits, labels):
    """ Loss function
    Args:
      logits: Logits
      labels: Targets

    Returns:
      Loss tensor.
    """
    # Calculate the cross entropy.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    # Calculate the average cross entropy
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss
