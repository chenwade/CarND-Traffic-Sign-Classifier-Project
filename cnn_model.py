from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Convolution2D, Dropout, BatchNormalization, Flatten, Dense, Input, Activation
from keras.models import Model

"""
implementation of LeNet as described on:

http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

replaced the subsampling layes with max pooling and used Xavier initialization
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten

import os
from preprocess import *


def conv2d(x, kernel_sz, depth, strides=1, padding='VALID'):
    weights = tf.get_variable('weights',
                              shape=[kernel_sz, kernel_sz, x.get_shape()[3], depth],
                              initializer=tf.glorot_normal_initializer())

    biases = tf.get_variable('biases', shape=[depth],
                             initializer=tf.constant_initializer(0.0))

    conv = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding=padding)
    return tf.nn.bias_add(conv, biases)


def fc(x, out_sz):
    weights = tf.get_variable('weights',
                              shape=[x.get_shape()[1], out_sz],
                              initializer=tf.glorot_normal_initializer())

    biases = tf.get_variable('biases', shape=[out_sz],
                             initializer=tf.constant_initializer(0.0))

    return tf.add(tf.matmul(x, weights), biases)


def maxpool2d(x, size=2, padding='VALID'):
    return tf.nn.max_pool(
        x,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding=padding)


def get_LeNet(features, n_classes):

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    with tf.variable_scope('conv1'):
        conv1 = conv2d(features, kernel_sz=5, depth=6)
        conv1 = tf.nn.relu(conv1)
        pool1 = maxpool2d(conv1)

    # Layer 2: Convolutional. Output = 10x10x16.
    with tf.variable_scope('conv2'):
        conv2 = conv2d(pool1, kernel_sz=5, depth=16)
        conv2 = tf.nn.relu(conv2)
        pool2 = maxpool2d(conv2)

    # Flatten. Input = 5x5x16. Output = 400.
    fc_input = flatten(pool2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    with tf.variable_scope('fc3'):
        fc3 = fc(fc_input, 120)
        fc3 = tf.nn.relu(fc3)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    with tf.variable_scope('fc4'):
        fc4 = fc(fc3, 84)
        fc4 = tf.nn.relu(fc4)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    with tf.variable_scope('out'):
        logits = fc(fc4, n_classes)

    return logits


def get_modern_cnn_keras(n_classes=10):

    img_input = Input(shape=(32, 32, 1), name='input_img')

    # Block 1:
    x = Convolution2D(32, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_1a')(img_input)
    x = BatchNormalization(name='bn_1a')(x)
    x = Activation('relu', name='relu_1a')(x)
    x = Convolution2D(32, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_1b')(x)
    x = BatchNormalization(name='bn_1b')(x)
    x = Activation('relu', name='relu_1b')(x)
    x = Convolution2D(32, 2, 2,
                      border_mode='valid',
                      activation='relu',
                      init='glorot_normal',
                      subsample=(2, 2),
                      name='conv_pool_1')(x)
    x = Dropout(0.2, name='dropout_1')(x)

    # Block 2:
    x = Convolution2D(64, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_2a')(x)
    x = BatchNormalization(name='bn_2a')(x)
    x = Activation('relu', name='relu_2a')(x)
    x = Convolution2D(64, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_2b')(x)
    x = BatchNormalization(name='bn_2b')(x)
    x = Activation('relu', name='relu_2b')(x)
    x = Convolution2D(64, 2, 2,
                      border_mode='valid',
                      activation='relu',
                      init='glorot_normal',
                      subsample=(2, 2),
                      name='conv_pool_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)

    # Block 3:
    x = Convolution2D(128, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_3a')(x)
    x = BatchNormalization(name='bn_3a')(x)
    x = Activation('relu', name='relu_3a')(x)
    x = Convolution2D(128, 3, 3,
                      border_mode='same',
                      activation='relu',
                      init='glorot_normal',
                      name='conv_3b')(x)
    x = BatchNormalization(name='bn_3b')(x)
    x = Activation('relu', name='relu_3b')(x)
    x = Convolution2D(128, 2, 2,
                      border_mode='valid',
                      activation='relu',
                      init='glorot_normal',
                      subsample=(2, 2),
                      name='conv_pool_3')(x)
    x = Dropout(0.2, name='dropout_3')(x)

    # Fully Connected:
    x = Flatten(name='flatten_4')(x)
    x = Dense(512, activation='relu', name='fc_6')(x)
    x = Dropout(0.5, name='dropout_7')(x)
    x = Dense(n_classes, activation='softmax', name='logits')(x)

    return Model(img_input, x, name='modern')


def get_modern_cnn(features, n_classes, dropout):
    """
       Performs a full model pass.

       Parameters
       ----------
       input         : Tensor
                       NumPy array containing a batch of examples.
       params        : Parameters
                       Structure (`namedtuple`) containing model parameters.
       dropout   : Tensor of type tf.bool
                       Flag indicating if we are training or not (e.g. whether to use dropout).

       Returns
       -------
       Tensor with predicted logits.
       """

    inputs = tf.reshape(features, [-1, 32, 32, 1])

    # Convolutions

    with tf.variable_scope('conv1'):
        conv1 = conv2d(inputs, kernel_sz=5, depth=32, strides=1, padding='SAME')
    with tf.variable_scope('pool1'):
        pool1 = maxpool2d(conv1, size=2)
        pool1 = tf.cond(dropout, lambda: tf.nn.dropout(pool1, keep_prob=0.2), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv2d(pool1, kernel_sz=5, depth=64, strides=1, padding='SAME')
    with tf.variable_scope('pool2'):
        pool2 = maxpool2d(conv2, size=2)
        pool2 = tf.cond(dropout, lambda: tf.nn.dropout(pool2, keep_prob=0.2), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv2d(pool2, kernel_sz=5, depth=128, strides=1, padding='SAME')
    with tf.variable_scope('pool3'):
        pool3 = maxpool2d(conv3, size=2)
        pool3 = tf.cond(dropout, lambda: tf.nn.dropout(pool3, keep_prob=0.2), lambda: pool3)

    # Fully connected, Combine 3 previous pool layer into one flattened layer
    # 1st stage output
    pool1 = tf.layers.max_pooling2d(pool1, pool_size=[4, 4], strides=4)
    pool1_shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, pool1_shape[1] * pool1_shape[2] * pool1_shape[3]])
    # 2nd stage output
    pool2 = tf.layers.max_pooling2d(pool2, pool_size=[2, 2], strides=2)
    pool2_shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
    # 3rd stage output
    pool3_shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, pool3_shape[1] * pool3_shape[2] * pool3_shape[3]])

    flattened = tf.concat([pool1, pool2, pool3], axis=1)

    # Fully connected layer
    with tf.variable_scope('fc4'):
        fc4 = fc(flattened, 1024)
        fc4 = tf.cond(dropout, lambda: tf.nn.dropout(fc4, keep_prob=0.5), lambda: fc4)
    # Output layer
    with tf.variable_scope('out'):
        logits = fc(fc4, n_classes)

    return logits


def get_modern_cnn1(features, labels, mode):
    '''
    This is a more new approach to convnets, featuring many of the state-of-the-art concepts
    introduced over the past few years.

    1 - Xavier weight initialization
    http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    2 - Batch Normalization
    https://arxiv.org/pdf/1502.03167.pdf

    3 - Dropout
    http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf
    '''


    # Input layer
    input_layer = tf.reshape(features, [-1, 32, 32, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 1]
    # Output Tensor Shape: [batch_size, 32, 32, 32]
    with tf.variable_scope('conv1'):
        conv_1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.glorot_normal_initializer())

    with tf.variable_scope('pool1'):
        pool1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=2)
        pool1 = tf.layers.dropout(pool1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 16, 16, 32]
    # Output Tensor Shape: [batch_size, 8, 8, 64]
    with tf.variable_scope('conv2'):
        conv_2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_normal_initializer())

    with tf.variable_scope('pool2'):
        pool2 = tf.layers.max_pooling2d(conv_2, pool_size=[2, 2], strides=2)
        pool2 = tf.layers.dropout(pool2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #3
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 8, 8, 64]
    # Output Tensor Shape: [batch_size, 4, 4, 128]
    with tf.variable_scope('conv3'):
        conv_3 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_normal_initializer())

    with tf.variable_scope('pool3'):
        pool3 = tf.layers.max_pooling2d(conv_3, pool_size=[2, 2], strides=2)
        pool3 = tf.layers.dropout(pool3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Combine 3 previous pool layer into one flattened layer
    pool1 = tf.layers.max_pooling2d(pool1, pool_size=[4, 4], strides=4)
    pool1_shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, pool1_shape[1] * pool1_shape[2] * pool1_shape[3]])
    pool2 = tf.layers.max_pooling2d(pool2, pool_size=[2, 2], strides=2)
    pool2_shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])

    pool3_shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, pool3_shape[1] * pool3_shape[2] * pool3_shape[3]])

    flattened = tf.concat([pool1, pool2, pool3], axis=1)

    # Fully Connected:
    # Input Tensor Shape: [batch_size, 3584]
    # Output Tensor Shape: [batch_size, 1000]
    with tf.variable_scope('fc4'):
        fc = tf.layers.dense(inputs=flattened, units=1000, activation=tf.nn.relu)
        fc_drop = tf.layers.dropout(inputs=fc, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Output layer
    # Input Tensor Shape: [batch_size, 1000]
    # Output Tensor Shape: [batch_size, 43]
    logits = tf.layers.dense(inputs=fc_drop, units=43)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



if __name__ == "__main__":

    import pickle
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

    X_train, y_train = train['features'], train['labels'].astype(np.int32)
    X_valid, y_valid = valid['features'], valid['labels'].astype(np.int32)
    X_test, y_test = test['features'], test['labels'].astype(np.int32)

    # from traffic_sign_classifier import get_preprocessed_data
    X_trn_pp, X_val_pp, X_tst_pp = get_preprocessed_data(X_train, X_valid, X_test)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=get_modern_cnn1, model_dir="/tmp/cnn_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X_trn_pp,
        y=y_train,
        batch_size=64,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X_val_pp,
        y=y_valid,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)








