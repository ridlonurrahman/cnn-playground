from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from train import *
from loss import *
from accuracy import *
from model import *

import cifar_input
import os.path
import time
import numpy as np
import tensorflow as tf

max_steps = 20000 # 312 Step per epoch
train_dir = 'cifar100_alexnet_model/'
batch_size = 128
log_device_placement = False

with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    # Generating images and its labels
    # build_inputs('cifar10/cifar100', cifar dataset dir, batch size, mode)
    images, labels = cifar_input.build_input('cifar100', '../../cifar/cifar100/train.bin', batch_size, 'train')
    
    # Creating graph. NUM_CLASESS=10 (CIFAR-10) or NUM_CLASESS=100 (CIFAR-100)
    logits = inference(images, NUM_CLASSES=100)
    
    # Loss/Error and Accuracy
    losses = loss(logits, labels)
    accuracies = accuracy(logits, labels)
    
    # Our train_op (Only minimizing loss)
    train_op = train(losses, global_step, batch_size)
    
    
    # SessionRunHook. Logging will be done each x steps.
    class _LoggerHook(tf.train.SessionRunHook):
        
      def begin(self):
        self._step = -1
        # Creating train_dir if it does not exist and writing to log file
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        open(train_dir+'training_data.csv', 'w').close()
        f = open(train_dir+"log.txt",'ab')
        f.write('\n\n==== Run ===\nInfo: Alexnet\n')
        f.close()

      def before_run(self, run_context):
        # Increment step, reset start_time, and asking for loss and accuracy tensor
        self._step += 1
        self._start_time = time.time()
        return tf.train.SessionRunArgs([losses, accuracies])

      def after_run(self, run_context, run_values):
        duration = time.time() - self._start_time # Calculating time
        loss_value = run_values.results[0]
        accuracy_value = run_values.results[1]
        # Printing log, accuracy, and loss
        if self._step % 10 == 0:
          print("{0}: step {1}, error = {2:.4f}, accuracy = {3:.4f}. ({4:.3f} sec/step)\n".format(
              datetime.now(), self._step, loss_value, accuracy_value, float(duration)))
                
          f = open(train_dir+"log.txt",'ab')
          f.write("{0}: step {1}, error = {2:.4f}, accuracy = {3:.4f}. ({4:.3f} sec/step)\n".format(
              datetime.now(), self._step, loss_value, accuracy_value, float(duration)))
          f.close()
                
          f = open(train_dir+"training_data.csv",'ab')
          f.write('{0},{1},{2}\n'.format(self._step, loss_value, accuracy_value))
          f.close()
            
    with tf.train.MonitoredTrainingSession(checkpoint_dir=train_dir,
                                           hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                                                  tf.train.NanTensorHook(losses),
                                                  _LoggerHook()],save_checkpoint_secs=30, 
                                           config=tf.ConfigProto(
                                               log_device_placement=log_device_placement)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)