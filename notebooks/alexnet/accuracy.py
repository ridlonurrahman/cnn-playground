import tensorflow as tf

def accuracy(logits, labels):
	"""Accuracy function
	Args:
	  logits: Logits
	  labels: Targets

	Returns:
	  Accuracy tensor.
	"""
	# Calculating accuracy from the logits outputs and targets
	return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
