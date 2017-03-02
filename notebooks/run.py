import os, time
import tensorflow as tf
import matplotlib.pyplot as plt

def run_training(init, train_data, valid_data, train_step, error, accuracy, inputs, targets, num_epoch=10):
    '''Run training'''
    err_train, acc_train, err_valid, acc_valid = {}, {}, {}, {}
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            start_time = time.time()
            running_error = 0.
            running_accuracy = 0.
            for input_batch, target_batch in train_data:
                _, batch_error, batch_acc = sess.run(
                    [train_step, error, accuracy], 
                    feed_dict={inputs: input_batch, targets: target_batch})
                running_error += batch_error
                running_accuracy += batch_acc
            running_error /= train_data.num_batches
            running_accuracy /= train_data.num_batches
            valid_error = 0.
            valid_accuracy = 0.               
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={inputs: input_batch, targets: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches

            err_train[epoch+1], acc_train[epoch+1], err_valid[epoch+1], acc_valid[epoch+1] = running_error, running_accuracy, valid_error, valid_accuracy
            total_time = time.time() - start_time
            if (epoch+1)%5 == 0:
                print('Epoch {0:02d} ({5:.2f}s): err(train)={1:.2f} acc(train)={2:.2f} err(valid)={3:.2f} acc(valid)={4:.2f}'.format(epoch + 1, running_error, running_accuracy, valid_error, valid_accuracy, total_time))
    return err_train, acc_train, err_valid, acc_valid

def run_training_with_bn(init, train_data, valid_data, train_step, error, accuracy, inputs, targets, num_epoch=10):
    '''Run training with batch normalisation. It needs the information whether the network is trained or not'''
    err_train, acc_train, err_valid, acc_valid = {}, {}, {}, {}
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            start_time = time.time()
            running_error = 0.
            running_accuracy = 0.
            for input_batch, target_batch in train_data:
                global phase_train 
                phase_train = 1
                _, batch_error, batch_acc = sess.run(
                    [train_step, error, accuracy], 
                    feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
                running_error += batch_error
                running_accuracy += batch_acc
            running_error /= train_data.num_batches
            running_accuracy /= train_data.num_batches
            valid_error = 0.
            valid_accuracy = 0.               
            for input_batch, target_batch in valid_data: 
                phase_train = 0
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches

            err_train[epoch+1], acc_train[epoch+1], err_valid[epoch+1], acc_valid[epoch+1] = running_error, running_accuracy, valid_error, valid_accuracy
            total_time = time.time() - start_time
            if (epoch+1)%5 == 0:
                print('Epoch {0:02d} ({5:.2f}s): err(train)={1:.2f} acc(train)={2:.4f} err(valid)={3:.2f} acc(valid)={4:.4f}'.format(epoch + 1, running_error, running_accuracy, valid_error, valid_accuracy, total_time))
    return err_train, acc_train, err_valid, acc_valid

def run_testing(init, train_data, valid_data, test_data, train_step, error, accuracy, inputs, targets, num_epoch=10):
    '''Run the training then the testing'''
    err_train, acc_train, err_valid, acc_valid = {}, {}, {}, {}
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            start_time = time.time()
            running_error = 0.
            running_accuracy = 0.
            for input_batch, target_batch in train_data:
                global phase_train 
                phase_train = 1
                _, batch_error, batch_acc = sess.run(
                    [train_step, error, accuracy], 
                    feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
                running_error += batch_error
                running_accuracy += batch_acc
            running_error /= train_data.num_batches
            running_accuracy /= train_data.num_batches
            valid_error = 0.
            valid_accuracy = 0.               
            for input_batch, target_batch in valid_data: 
                phase_train = 0
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches

            err_train[epoch+1], acc_train[epoch+1], err_valid[epoch+1], acc_valid[epoch+1] = running_error, running_accuracy, valid_error, valid_accuracy
            total_time = time.time() - start_time
            if (epoch+1)%5 == 0:
                print('Epoch {0:02d} ({5:.2f}s): err(train)={1:.2f} acc(train)={2:.4f} err(valid)={3:.2f} acc(valid)={4:.4f}'.format(epoch + 1, running_error, running_accuracy, valid_error, valid_accuracy, total_time))
        test_error = 0.
        test_accuracy = 0.               
        for input_batch, target_batch in test_data: 
            phase_train = 0
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
            test_error += batch_error
            test_accuracy += batch_acc
        test_error /= test_data.num_batches
        test_accuracy /= test_data.num_batches
        print('Error testing = {0:.2f} Accuracy testing = {1:.4f}'.format(test_error, test_accuracy))
    return err_train, acc_train, err_valid, acc_valid

def run_testing_with_early_stopping(init, train_data, valid_data, test_data, train_step, error, accuracy, inputs, targets, num_epoch=10):
    '''Run the training with modified early stopping then the testing'''
    err_train, acc_train, err_valid, acc_valid = {}, {}, {}, {}
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            start_time = time.time()
            running_error = 0.
            running_accuracy = 0.
            for input_batch, target_batch in train_data:
                global phase_train 
                phase_train = 1
                _, batch_error, batch_acc = sess.run(
                    [train_step, error, accuracy], 
                    feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
                running_error += batch_error
                running_accuracy += batch_acc
            running_error /= train_data.num_batches
            running_accuracy /= train_data.num_batches
            valid_error = 0.
            valid_accuracy = 0.               
            for input_batch, target_batch in valid_data: 
                phase_train = 0
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches

            err_train[epoch+1], acc_train[epoch+1], err_valid[epoch+1], acc_valid[epoch+1] = running_error, running_accuracy, valid_error, valid_accuracy
            total_time = time.time() - start_time
            if (epoch+1)%5 == 0:
                print('Epoch {0:02d} ({5:.2f}s): err(train)={1:.2f} acc(train)={2:.4f} err(valid)={3:.2f} acc(valid)={4:.4f}'.format(epoch + 1, running_error, running_accuracy, valid_error, valid_accuracy, total_time))
        
        early_stopping_epoch = max(acc_valid, key=acc_valid.get)
        print("Early stopping at = {0} epochs".format(early_stopping_epoch))
        sess.run(init)
        for epoch in range(early_stopping_epoch):
            start_time = time.time()
            running_error = 0.
            running_accuracy = 0.
            for input_batch, target_batch in train_data:
                phase_train = 1
                _, batch_error, batch_acc = sess.run(
                    [train_step, error, accuracy], 
                    feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
                running_error += batch_error
                running_accuracy += batch_acc
            running_error /= train_data.num_batches
            running_accuracy /= train_data.num_batches
        
        test_error = 0.
        test_accuracy = 0.               
        for input_batch, target_batch in test_data: 
            phase_train = 0
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={'inputs:0': input_batch, 'targets:0': target_batch})
            test_error += batch_error
            test_accuracy += batch_acc
        test_error /= test_data.num_batches
        test_accuracy /= test_data.num_batches
        print('Error testing = {0:.2f} Accuracy testing = {1:.4f}'.format(test_error, test_accuracy))
    return err_train, acc_train, err_valid, acc_valid
