from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import six

import numpy as np
import tensorflow as tf

import cifar_input
from train import *
from loss import *
from accuracy import *
from model import *


train_dir = 'cifar100_alexnet_model/'
batch_size = 128
log_device_placement = False

def evaluate():
  eval_batch_count = 50 # faster validation, change to 40000/batch_size for the report
  validation_error=0
  validation_accuracy=0

  with tf.device('/cpu:0'):
      with tf.Graph().as_default() as g:
        images, labels = cifar_input.build_input('cifar100', '../../cifar/cifar100/validation.bin', batch_size, 'eval') #TEST.BIN OR VALIDATION.BIN

        logits = inference(images, NUM_CLASSES=100)
        saver = tf.train.Saver()
        
        losses = loss(logits, labels)
    
        accuracies = accuracy(logits, labels)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        tf.train.start_queue_runners(sess)

        ckpt_state = tf.train.get_checkpoint_state(train_dir)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        for _ in six.moves.range(eval_batch_count):
          (value_losses, value_accuracy) = sess.run(
              [losses, accuracies])
          validation_error += value_losses
          validation_accuracy += value_accuracy
        validation_error /= eval_batch_count
        validation_accuracy /= eval_batch_count
        
        step = str(ckpt_state.model_checkpoint_path).split('-')[1]
        tf.logging.info('loss: %.3f, best accuracy: %.3f' %
                        (validation_error, validation_accuracy))
        f = open(train_dir+"validation_data.csv",'ab')
        f.write('{0},{1},{2}\n'.format(step, validation_error,validation_accuracy))
        f.close()
        f = open(train_dir+"log.txt",'ab')
        f.write('loss: {0}, best accuracy: {1}\n'.format(validation_error, validation_accuracy))
        f.close()

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
open(train_dir+'validation_data.csv', 'w').close()
while True:
    evaluate()
    time.sleep(5)