# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import os
import seaborn as sns
import pandas as pd
import collections
import random

from importlib import reload
from preprocess import *
from skimage import exposure
from keras.utils import np_utils
from sklearn.utils import shuffle
from cnn_model import *
from helper import *

import sys
import time

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def evaluate_model(X_data, y_data, model_name, batch_size=64):

    assert len(X_data) == len(y_data)
    # ## construct tensorflow graph ##
    # ----------------
    lenet_graph = tf.Graph()
    with lenet_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, n_classes)

        #is_dropout = tf.placeholder(tf.bool)
        logits = get_LeNet(x, n_classes=n_classes)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # evaluate the model by using test data
    num_examples = len(X_data)
    total_accuracy = 0
    with tf.Session(graph=lenet_graph) as sess:
        saver = tf.train.Saver()
        try:
            saver.restore(sess, 'models/' + model_name)
        except:
            print("Failed restoring previously trained model: file does not exist.")
            pass

        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * batch_size)

        accuracy = total_accuracy / num_examples
        print("================= Evaluate ==================")
        print("Model evaluate accuracy = %.4f" % accuracy)


def visualize_data(y_train, y_valid, y_test):
    n_classes = len(np.unique(y_train))
    # We will load the sign names from the provided csv file and add a counter to each sign class
    sign_names = pd.read_csv('signnames.csv')
    # Add counter
    df2 = pd.DataFrame.from_dict(collections.Counter(y_train), orient='index').reset_index()
    df2 = df2.rename(columns={'index': 'ClassId', 0: 'Count'})
    sign_names['NumImages'] = df2['Count']

    plt.figure(figsize=(20, 4))
    sns.barplot(sign_names['ClassId'].values, sign_names['NumImages'].values)
    plt.xlabel('Sign Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.show()

    train_counter = collections.Counter(y_train)
    train_counter = [train_counter[i] for i in range(n_classes)]
    train_percentage = train_counter / np.sum(train_counter)
    valid_counter = collections.Counter(y_valid)
    valid_counter = [valid_counter[i] for i in range(n_classes)]
    valid_percentage = valid_counter / np.sum(valid_counter)
    test_counter = collections.Counter(y_test)
    test_counter = [test_counter[i] for i in range(n_classes)]
    test_percentage = test_counter / np.sum(test_counter)

    bar_width = 0.2
    index = np.arange(n_classes)
    fig, ax = plt.subplots(figsize=(20, 10))

    train_rects = ax.bar(index, train_percentage, bar_width, color='r', label='train_set')
    valid_rects = ax.bar(index + bar_width, valid_percentage, bar_width, color='g', label='valid_set')
    test_rects = ax.bar(index + 2 * bar_width, test_percentage, bar_width, color='b', label='test_set')
    ax.set_title('Distribution of traffic signs in trianing/validation/test set')
    ax.set_ylabel('Percentage')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(sign_names.SignName)
    # reverse x label
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment('right')

    ax.legend()
    fig.tight_layout()
    plt.show()

    # visualize all classes
    implot, axes = plt.subplots(5, 9, figsize=(25, 15))
    for classid in range(n_classes):
        X_class = X_train[y_train == classid]
        rnd_idx = random.sample(range(len(X_class)), 1)

        row = classid // 9
        col = classid % 9
        axes[row, col].imshow(X_class[rnd_idx[0]])
        axes[row, col].grid(False)
        axes[row, col].axis('off')
        axes[row, col].set_title(sign_names.SignName[classid])
    plt.show()

    # visualize the 10 class which have least data samples
    bottom10 = sign_names.sort_values(by=['NumImages'])[0: 10]['ClassId'].values

    for classid in bottom10:
        print(sign_names.SignName[classid], ':')
        implot = plt.figure(figsize=(12, 1))
        X_class = X_train[y_train == classid]
        rnd_idx = random.sample(range(len(X_class)), 10)
        for i in range(10):
            ax = implot.add_subplot(1, 10, i + 1)
            ax.grid(False)
            ax.axis('off')
            ax.imshow(X_class[rnd_idx[i]])
        plt.show()

    # visualize the 10 class which have least data samples
    top10 = sign_names.sort_values(by=['NumImages'], ascending=False)[0: 10]['ClassId'].values

    for classid in top10:
        print(sign_names.SignName[classid], ':')
        implot = plt.figure(figsize=(12, 1))
        X_class = X_train[y_train == classid]
        rnd_idx = random.sample(range(len(X_class)), 10)
        for i in range(10):
            ax = implot.add_subplot(1, 10, i + 1)
            ax.grid(False)
            ax.axis('off')
            ax.imshow(X_class[rnd_idx[i]])
        plt.show()


def lenet_modeling(X_train, y_train, X_valid, y_valid, X_test, y_test, params):

    lr, max_epoches, batch_size, l2_lambda, model_name, resume_training, early_stopping_patience, epochs_print = params
    n_classes = len(np.unique(y_train))

    start = time.time()
    # ## construct tensorflow graph ##
    # ----------------
    lenet_graph = tf.Graph()
    with lenet_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, n_classes)

        #is_dropout = tf.placeholder(tf.bool)
        logits = get_LeNet(x, n_classes=n_classes)
        logits_probs = tf.nn.softmax(logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        with tf.variable_scope('fc3', reuse=True):
            reg_fc3 = tf.nn.l2_loss(tf.get_variable('weights'))

        with tf.variable_scope('fc4', reuse=True):
            reg_fc4 = tf.nn.l2_loss(tf.get_variable('weights'))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
        # L2 regularize
        loss_operation = tf.reduce_mean(cross_entropy + l2_lambda * (reg_fc3 + reg_fc4))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        training_operation = optimizer.minimize((loss_operation))

    # ##  train models ## #
    # ---------------------
    with tf.Session(graph=lenet_graph) as sess:
        sess.run(tf.global_variables_initializer())

        def acc_and_loss(X_data, y_data, batch_size=64):
            num_examples = len(X_data)
            total_accuracy = 0
            softmax_cross_entropy = []
            sess = tf.get_default_session()
            for offset in range(0, num_examples, batch_size):
                batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
                [accuracy, softmax] = sess.run([accuracy_operation, cross_entropy],
                                               feed_dict={x: batch_x, y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
                softmax_cross_entropy.extend(softmax)
            return total_accuracy / num_examples, np.mean(softmax_cross_entropy)

        saver = tf.train.Saver()
        # If we chose to keep training previously trained model, restore session.
        if resume_training:
            try:
                tf.train.Saver().restore(sess,'models/' + model_name)
            except Exception as e:
                print(e)
                print("Failed restoring previously trained model: file does not exist.")
                pass
        else:
            print("Training new model...")

        early_stopping = EarlyStopping(saver, sess, model_name, patience=early_stopping_patience, minimize=True)

        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)

        train_num = len(y_train)
        for i in range(max_epoches):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, train_num, batch_size):
                batch_x = X_train[offset: offset + batch_size]
                batch_y = y_train[offset: offset + batch_size]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            training_acc, training_loss = acc_and_loss(X_train, y_train)
            valid_acc, valid_loss = acc_and_loss(X_valid, y_valid)

            train_accuracy_history = np.append(train_accuracy_history, training_acc)
            train_loss_history = np.append(train_loss_history, training_loss)
            valid_accuracy_history = np.append(valid_accuracy_history, valid_acc)
            valid_loss_history = np.append(valid_loss_history, valid_loss)

            if epochs_print > 0:
                if (i % epochs_print == 0):
                    print("EPOCH: {}, lr: {:.5f} ...".format(i, lr))
                    print("Training Loss = {:.5f}, Training Accuracy = {:.3f}".format(training_loss, training_acc))
                    print("Validation Loss = {:.5f}, Validation Accuracy = {:.3f}".format(valid_loss, valid_acc))
                    print()

            if early_stopping_patience > 0:
                if early_stopping(valid_loss, i):
                    print("Early stopping.\nBest monitored loss was {:.6f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    break


        ### evaluate model ###
        test_acc, test_loss = acc_and_loss(X_test, y_test)
        valid_acc, valid_loss = acc_and_loss(X_valid, y_valid)
        print("=============================================")
        print(" Valid loss: %.8f, accuracy = %.4f)" % (valid_loss, valid_acc))
        print(" Test loss: %.8f, accuracy = %.4f)" % (test_loss, test_acc))
        print(" Total time: " + get_time_hhmmss(start))
        print("  Timestamp: " + get_time_hhmmss())

        # store the training/validation history
        np.savez('models/' + model_name + '_history',
                 train_loss_history=train_loss_history,
                 train_accuracy_history=train_accuracy_history,
                 valid_loss_history=valid_loss_history,
                 valid_accuracy_history=valid_accuracy_history)

        print("Model saved")
        plot_history(model_name)


if __name__ == '__main__':

    ###  LOAD DATASET  ###
    path = ''
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_traffic_sign_data(path)

    n_train = len(y_train)
    n_valid = len(y_valid)
    n_test = len(y_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))

    print("Number of training examples =", n_train)
    print("Number of valid examples =", n_valid)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    #### Visualize the data ####
    # visualize_data(X_train, y_train, X_valid, y_valid, X_test, y_test)

    #### Preprocess the data  ####
    # augment the training dataset 7 times
    X_trn_pp, y_trn_pp = preprocess_data(X_train, y_train, dataset="train", augment=7, preprocess=True)
    X_val_pp, y_val_pp = preprocess_data(X_valid, y_valid, dataset="valid", augment=1, preprocess=True)
    X_tst_pp, y_tst_pp = preprocess_data(X_test, y_test, dataset="test", augment=1, preprocess=True)

    # show the preprocessed images
    implot = plt.figure(figsize=(12, 4))
    rnd_idx = random.sample(range(len(X_train)), 10)
    for i in range(10):
        ax1 = implot.add_subplot(2, 10, i + 1)
        ax1.grid(False)
        ax1.axis('off')
        ax1.imshow(X_train[rnd_idx[i]])

        ax2 = implot.add_subplot(1, 10, i + 1)
        ax2.grid(False)
        ax2.axis('off')
        ax2.imshow(img_preprocess(X_train[rnd_idx[i]])[:, :, 0], cmap='gray')
    plt.show()

    #### Normalize the data set ####
    mean = np.mean(X_trn_pp)
    stdd = np.std(X_trn_pp)
    scaled_X_trn_pp = (X_trn_pp - mean) / stdd
    scaled_X_val_pp = (X_val_pp - mean) / stdd
    scaled_X_tst_pp = (X_tst_pp - mean) / stdd


    #### Control panel 1, setup model parameters ####
    model_name = 'lenet_'

    max_epoches = 300
    batch_size = 64
    epochs_print = 5

    lr = 1e-3
    l2_lambda = 1e-3

    resume_training = False
    early_stopping_patience = 50

    params = [lr, max_epoches, batch_size, l2_lambda, model_name,
             resume_training, early_stopping_patience, epochs_print]

    #### train the model and evaluate the mode ###
    lenet_modeling(scaled_X_trn_pp, y_trn_pp, scaled_X_val_pp, y_val_pp, scaled_X_tst_pp, y_tst_pp, params=params)

    #### Control panel 2, setup model parameters ####
    model_name = 'lenet_'

    max_epoches = 300
    batch_size = 64
    epochs_print = 5

    lr = 1e-5
    l2_lambda = 1e-3

    resume_training = True
    early_stopping_patience = 50

    params = [lr, max_epoches, batch_size, l2_lambda, model_name,
              resume_training, early_stopping_patience, epochs_print]

    #### Modify learning rate and fine tuning the model ###
    lenet_modeling(scaled_X_trn_pp, y_trn_pp, scaled_X_val_pp, y_val_pp, scaled_X_tst_pp, y_tst_pp, params=params)

    #### Evaluate the trained model ####
    evaluate_model(scaled_X_tst_pp, y_tst_pp, model_name)


