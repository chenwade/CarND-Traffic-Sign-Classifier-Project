# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
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
from lenet_bg import *
from helper import *


# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def evaluate(X_data, y_data):
    BATCH_SIZE = 64
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset: offset + BATCH_SIZE], y_data[offset: offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y:batch_y})
        total_accuracy += (accuracy * BATCH_SIZE)
    return total_accuracy / num_examples


def acc_and_loss(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    softmax_cross_entropy = []
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        [accuracy, softmax] = sess.run([accuracy_operation, cross_entropy], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        softmax_cross_entropy.extend(softmax)
    return total_accuracy / num_examples, np.mean(softmax_cross_entropy)


def train_nn(X_tr, y_tr, X_vl, y_vl, params):
    # get the params
    lr, EPOCHS, BATCH_SIZE, l2_lambda, model_name, resume_training, early_stopping_patience, epochs_print = params

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_tr)

        if resume_training:
            try:
                saver.restore(sess, 'models/' + model_name)
                print('Resuming previous training...')
            except:
                print("Training new model...")
        print()

        early_stopping = EarlyStopping(saver, sess, patience=early_stopping_patience, minimize=True)

        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)

        for i in range(EPOCHS):
            X_tr, y_tr = shuffle(X_tr, y_tr)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_tr[offset: offset+BATCH_SIZE],  y_tr[offset: offset+BATCH_SIZE]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            valid_accuracy, valid_loss = acc_and_loss(X_vl, y_vl)
            train_accuracy, train_loss = acc_and_loss(X_tr, y_tr)

            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])

            if epochs_print > 0:
                if (i % epochs_print == 0):
                    print("EPOCH: {}, lr: {:.5f} ...".format(i, lr))
                    print("Training Loss = {:.5f}, Training Accuracy = {:.3f}".format(train_loss, train_accuracy))
                    print("Validation Loss = {:.5f}, Validation Accuracy = {:.3f}".format(valid_loss, valid_accuracy))
                    print()

            if early_stopping_patience > 0:
                if early_stopping(valid_loss, i):
                    print("Early stopping.\nBest monitored loss was {:.6f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    break

        np.savez('models/' + model_name + '_history',
                 train_loss_history=train_loss_history,
                 train_accuracy_history=train_accuracy_history,
                 valid_loss_history=valid_loss_history,
                 valid_accuracy_history=valid_accuracy_history)

        print("Model saved")


def plot_history(model_name):
    history = np.load('models/' + model_name + '_history.npz')

    implot = plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss_history'], '-o')
    plt.plot(history['valid_loss_history'], '-o')
    plt.legend(['trn_loss', 'val_loss'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy_history'], '-o')
    plt.plot(history['valid_accuracy_history'], '-o')
    plt.legend(['trn_accuracy', 'val_accuracy'], loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()


def read_all_imgs(path, color=cv2.IMREAD_COLOR):
    images = []
    filenames = []

    filelist = os.listdir(path)

    for file in filelist:

        try:
            img = cv2.imread(path + file, color)
        except:
            img = None

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            images.append(img)
            filenames.append(file)

    return images, filenames


if __name__ == '__main__':

    path = ''
    training_file = path + 'train.p'
    validation_file = path + 'valid.p'
    testing_file = path + 'test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    ### Replace each question mark with the appropriate value.
    ### Use python, pandas or numpy methods rather than hard coding the results

    # TODO: Number of training examples
    assert (len(X_train) == len(y_train))
    assert (len(X_valid) == len(y_valid))
    assert (len(X_test) == len(y_test))
    n_train = len(X_train)
    n_valid = len(X_valid)
    n_test = len(X_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))

    print("Number of training examples =", n_train)
    print("Number of valid examples =", n_valid)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)


    # We will load the sign names from the provided csv file and add a counter to each sign class
    sign_names = pd.read_csv('signnames.csv')
    #Add counter
    a = collections.Counter(y_train)
    b = pd.DataFrame.from_dict(a, orient='index')
    df2 = pd.DataFrame.from_dict(collections.Counter(y_train), orient='index').reset_index()
    df2 = df2.rename(columns={'index': 'ClassId', 0: 'Count'})
    sign_names['NumImages'] = df2['Count']

    """
    plt.figure(figsize=(20, 4))
    sns.barplot(sign_names['ClassId'].values, sign_names['NumImages'].values, alpha=0.8, )
    plt.xlabel('Sign Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.show()

    bottom10 = sign_names.sort_values(by=['NumImages'])[0: 10]['ClassId'].values

    for classid in bottom10:
        print(sign_names.SignName[classid], ':')
        implot = plt.figure(figsize=(12, 1))
        X_class = X_train[y_train==classid]
        rnd_idx = random.sample(range(len(X_class)), 10)
        for i in range(10):
            ax = implot.add_subplot(1, 10, i + 1)
            ax.grid(False)
            ax.axis('off')
            ax.imshow(X_class[rnd_idx[i]])
        plt.show()

    top10 = sign_names.sort_values(by=['NumImages'], ascending=False)[0: 10]['ClassId'].values

    for classid in top10:
        print(sign_names.SignName[classid], ':')
        implot = plt.figure(figsize=(12, 1))
        X_class = X_train[y_train==classid]
        rnd_idx = random.sample(range(len(X_class)), 10)
        for i in range(10):
            ax = implot.add_subplot(1, 10, i + 1)
            ax.grid(False)
            ax.axis('off')
            ax.imshow(X_class[rnd_idx[i]])
        plt.show()
    

    # X_trn_pp = preprocess_array(X_train)
    # X_val_pp = preprocess_array(X_valid)

    # save_array('models/X_train_preprocessed.dat', X_trn_pp)
    # save_array('models/X_valid_preprocessed.dat', X_val_pp)

    X_trn_pp = load_array('models/X_train_preprocessed.dat')
    X_val_pp = load_array('models/X_valid_preprocessed.dat')

    y_trn_hot = np_utils.to_categorical(y_train)
    y_val_hot = np_utils.to_categorical(y_valid)

    assert X_trn_pp.shape[0] == y_trn_hot.shape[0]
    assert X_val_pp.shape[0] == y_val_hot.shape[0]

    #pp_mean = np.mean(X_trn_pp)
    #X_trn_pp -= pp_mean
    #X_val_pp -= pp_mean

    X_trn_pp, y_train = shuffle(X_trn_pp, y_train)
    y_trn_hot = np_utils.to_categorical(y_train)
    """

    """
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
    """


    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)
    logits = Get_LeNet(x, n_classes=n_classes)
    logits_probs = tf.nn.softmax(logits)

    l2_lambda = 1e-3
    lr = 1e-3

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    with tf.variable_scope('fc3', reuse=True):
        reg_fc3 = tf.nn.l2_loss(tf.get_variable('weights'))

    with tf.variable_scope('fc4', reuse=True):
        reg_fc4 = tf.nn.l2_loss(tf.get_variable('weights'))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    # L2 regularize
    loss_operation = tf.reduce_mean(cross_entropy) + l2_lambda / 2 * (reg_fc3 + reg_fc4)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    training_operation = optimizer.minimize((loss_operation))
    saver = tf.train.Saver()


    """
    # #### CONTROL PANEL 1 #### #

    model_name = 'lenet_'

    EPOCHS = 200
    BATCH_SIZE = 64

    lr = 1e-3
    l2_lambda = 1e-3

    resume_training = False

    early_stopping_patience = 50
    epochs_print = 10

    X_tr = X_trn_pp
    y_tr = y_train
    X_vl = X_val_pp
    y_vl = y_valid

    param = [lr, EPOCHS, BATCH_SIZE, l2_lambda, model_name,
             resume_training, early_stopping_patience, epochs_print]

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    train_nn(X_trn_pp, y_train, X_val_pp, y_valid, param)
    plot_history(model_name)
    a = 1
    """

    """
    # augument data
    rnd_idx = random.sample(range(len(X_train)), 5)

    fig, axes = plt.subplots(5, 5, figsize=(8, 15))
    print("From Left to right: Original Img, Histogram Equalization, Rotate, Motion Blur and Affine")
    for i in range(5):
        axes[i, 0].grid(False)
        axes[i, 0].axis('off')
        axes[i, 0].imshow(X_train[rnd_idx[i]])

        axes[i, 1].grid(False)
        axes[i, 1].axis('off')
        axes[i, 1].imshow(Adapthisteq(X_train[rnd_idx[i]]))

        axes[i, 2].grid(False)
        axes[i, 2].axis('off')
        axes[i, 2].imshow(Rotate(Adapthisteq(X_train[rnd_idx[i]])))

        axes[i, 3].grid(False)
        axes[i, 3].axis('off')
        axes[i, 3].imshow(Motionblur(Adapthisteq(X_train[rnd_idx[i]])))

        axes[i, 4].grid(False)
        axes[i, 4].axis('off')
        axes[i, 4].imshow(Affine(Adapthisteq(X_train[rnd_idx[i]])))
    plt.show()
    
    

    # X_trn_aug, y_trn_aug = Augment_dataset(X_trn_pp, y_train, augs=6)
    # X_trn_aug, y_trn_aug = shuffle(X_trn_aug, y_trn_aug)

    # save_array('models/X_train_augmented.dat', X_trn_aug)
    # save_array('models/y_train_augmented.dat', y_trn_aug)

    X_trn_aug = load_array('models/X_train_augmented.dat')
    y_trn_aug = load_array('models/y_train_augmented.dat')

    y_trn_aug_hot = np_utils.to_categorical(y_trn_aug)

    X_tr = X_trn_aug
    y_tr = y_trn_aug
    X_vl = X_val_pp
    y_vl = y_valid


    # #### CONTROL PANEL #### #

    model_name = 'lenet_aug'

    EPOCHS = 50
    BATCH_SIZE = 64

    lr = 1e-3
    l2_lambda = 1e-4

    resume_training = False

    early_stopping_patience = 25
    epochs_print = 1

    X_tr = X_trn_aug
    y_tr = y_trn_aug
    X_vl = X_val_pp
    y_vl = y_valid

    param = [lr, EPOCHS, BATCH_SIZE, l2_lambda, model_name,
             resume_training, early_stopping_patience, epochs_print]

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    loss_operation = tf.reduce_mean(cross_entropy) + l2_lambda / 2 * (reg_fc3 + reg_fc4)

    train_nn(X_tr, y_tr, X_vl, y_vl, param)

    plot_history(model_name)
    """

    X_val_pp = load_array('models/X_valid_preprocessed.dat')
    y_val_hot = np_utils.to_categorical(y_valid)

    X_vl = X_val_pp
    y_vl = y_valid

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('models/'))

        test_accuracy = evaluate(X_vl, y_vl)
        print("Validation Accuracy = {:.3f}".format(test_accuracy))


    model_name = 'lenet_aug'

    EPOCHS = 50
    BATCH_SIZE = 64

    lr = 1e-3
    l2_lambda = 1e-4

    resume_training = False

    early_stopping_patience = 25
    epochs_print = 1

    new_img, new_img_filename = read_all_imgs('input/')
    new_img = np.array(new_img)

    implot = plt.figure(figsize=(12, 8))
    for i in range(5):
        ax1 = implot.add_subplot(2, 5, i + 1)
        ax1.grid(False)
        ax1.axis('off')
        ax1.imshow(new_img[i])

        ax2 = implot.add_subplot(1, 5, i + 1)
        ax2.grid(False)
        ax2.axis('off')
        ax2.imshow(img_preprocess(new_img[i])[:, :, 0], cmap='gray')
    plt.show()

    model_name = 'lenet_aug'

    X_test_new = preprocess_array(new_img)
    num_images = len(X_test_new)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('models/'))
        # saver.restore(sess, 'models/' + 'early_stopping_checkpoint')
        logits, logits_probs = sess.run([logits, logits_probs], feed_dict={x: X_test_new})

        top5 = sess.run(tf.nn.top_k(tf.constant(logits_probs), k=5))

        test_new_class = np.argmax(logits, axis=1)

        for i in range(num_images):
            print(sign_names.SignName[test_new_class[i]], ':')
            plot_img(new_img[i])
            plt.show()

        for i in range(num_images):
            top5_classes = np.array(sign_names.SignName[top5[1][i, :]])
            top5_probs = np.array(top5[0][i, :])

            implot = plt.figure(figsize=(10, 4))

            ax = implot.add_subplot(1, 2, 1)
            ax.grid(False)
            ax.axis('off')
            ax.imshow(new_img[i])

            # classes = test_names
            # probs = test_probs
            y_pos = np.arange(len(top5_classes))

            ax2 = implot.add_subplot(1, 2, 2)
            ax2.barh(y_pos, top5_probs, align='center')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top5_classes)
            ax2.yaxis.set_ticks_position('right')
            ax2.set_xlabel('Top 5 Probabilities')
            ax2.set_title('Classes')
            ax2.set_xlim([0, 1])

            plt.tight_layout()

            plt.show()





    a = 1

