# Load pickled data
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import tensorflow as tf

import seaborn as sns
import pandas as pd
import collections
import random

from sklearn.utils import shuffle
from cnn_model import *
from helper import *
from sklearn.metrics import confusion_matrix

tf.logging.set_verbosity(tf.logging.INFO)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def visualize_data(X_train, y_train, X_valid, y_valid, X_test, y_test):

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
        X_class = X_train[y_train==classid]
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
        X_class = X_train[y_train==classid]
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
        X_class = X_train[y_train==classid]
        rnd_idx = random.sample(range(len(X_class)), 10)
        for i in range(10):
            ax = implot.add_subplot(1, 10, i + 1)
            ax.grid(False)
            ax.axis('off')
            ax.imshow(X_class[rnd_idx[i]])
        plt.show()


def modern_cnn_modeling(X_train, y_train, X_valid, y_valid, X_test, y_test, params):

    lr, max_epoches, batch_size, l2_lambda, model_name, resume_training, early_stopping_patience, epochs_print = params
    n_classes = len(np.unique(y_train))

    start = time.time()
    # ## construct tensorflow graph ##
    # ----------------
    modern_cnn_graph = tf.Graph()
    with modern_cnn_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, n_classes)

        is_dropout = tf.placeholder(tf.bool)
        logits = get_modern_cnn(x, n_classes=n_classes, dropout=is_dropout)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        with tf.variable_scope('fc4', reuse=True):
            reg_fc4 = tf.nn.l2_loss(tf.get_variable('weights'))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
        # L2 regularize
        loss_operation = tf.reduce_mean(cross_entropy + l2_lambda * reg_fc4)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        training_operation = optimizer.minimize(loss_operation)

    # ##  train models ## #
    # ---------------------
    with tf.Session(graph=modern_cnn_graph) as sess:
        sess.run(tf.global_variables_initializer())

        def acc_and_loss(X_data, y_data, batch_size=64):
            num_examples = len(X_data)
            total_accuracy = 0
            softmax_cross_entropy = []
            sess = tf.get_default_session()
            for offset in range(0, num_examples, batch_size):
                batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
                [accuracy, softmax] = sess.run([accuracy_operation, cross_entropy],
                                               feed_dict={x: batch_x, y: batch_y, is_dropout: False})
                total_accuracy += (accuracy * len(batch_x))
                softmax_cross_entropy.extend(softmax)
            return total_accuracy / num_examples, np.mean(softmax_cross_entropy)

        saver = tf.train.Saver()
        # If we chose to keep training previously trained model, restore session.
        if resume_training:
            try:
                tf.train.Saver().restore(sess, 'models/' + model_name)
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
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_dropout: False})

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


        ### Evaluate model ###
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
        # plot_history(model_name)



def evaluate_model(X_data, y_data, model_name, batch_size=64):

    assert len(X_data) == len(y_data)
    n_classes = len(np.unique(y_data))
    # ## construct tensorflow graph ##
    # ----------------
    modern_cnn_graph = tf.Graph()
    with modern_cnn_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, n_classes)

        is_dropout = tf.placeholder(tf.bool)
        logits = get_modern_cnn(x, n_classes=n_classes, dropout=is_dropout)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # evaluate the model by using test data
    num_examples = len(X_data)
    total_accuracy = 0
    with tf.Session(graph=modern_cnn_graph) as sess:
        saver = tf.train.Saver()
        try:
            saver.restore(sess, 'models/' + model_name)
        except:
            print("Failed restoring previously trained model: file does not exist.")
            pass

        predict_y = np.empty([0], dtype=np.int32)
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, is_dropout: False})
            total_accuracy += (accuracy * batch_size)

        accuracy = total_accuracy / num_examples
        print("================= Evaluate ==================")
        print("Model evaluate accuracy = %.4f" % accuracy)


def failure_analyse(X_data, y_data, model_name, batch_size=64):

    assert len(X_data) == len(y_data)
    n_classes = len(np.unique(y_data))
    # ## construct tensorflow graph ##
    # ----------------
    modern_cnn_graph = tf.Graph()
    with modern_cnn_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))

        is_dropout = tf.placeholder(tf.bool)
        logits = get_modern_cnn(x, n_classes=n_classes, dropout=is_dropout)
        prediction = tf.argmax(logits, 1)

    # evaluate the model by using test data
    num_examples = len(X_data)
    with tf.Session(graph=modern_cnn_graph) as sess:
        saver = tf.train.Saver()
        try:
            saver.restore(sess, 'models/' + model_name)
        except:
            print("Failed restoring previously trained model: file does not exist.")
            pass

        predict_y = np.empty([0], dtype=np.int32)
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
            pred_batch = sess.run(prediction, feed_dict={x: batch_x, y: batch_y, is_dropout: False})

            predict_y = np.append(predict_y, pred_batch)

    ###  Analyse the incorrect recognize images ####
    incorrect_index = np.nonzero(predict_y != y_data)[0]
    print("Total test samples = %d, unrecognized test samples = %d " % (num_examples, len(incorrect_index)))

    incorrect_classes = [y_data[i] for i in incorrect_index]

    df2 = pd.DataFrame.from_dict(collections.Counter(incorrect_classes), orient='index')
    df2 = df2.rename(columns={'index': 'ClassId', 0: 'Incorrect'})

    sign_names = pd.read_csv('signnames.csv')
    sign_names['Incorrect'] = df2['Incorrect']
    sign_names.fillna(value=0, inplace=True)

    index = np.arange(n_classes)
    sns.barplot(sign_names['SignName'].values, sign_names['Incorrect'].values, alpha=0.8, figsize=(20, 12))
    plt.xticks(index, sign_names['SignName'].values, rotation=90)

    plt.title('Incorrect number of Sign Classes', fontsize=12)
    plt.ylabel('Incorrect Classification', fontsize=12)
    plt.subplots_adjust(bottom=0.5)
    plt.show()

    # Using confusion matrix to analyse the incorrect data
    cm = confusion_matrix(y_data, predict_y)
    plot_confusion_matrix(cm, sign_names['ClassId'].values)

    k = 5
    classses, confuse_classes = find_topk_confusion(cm, 5)

    testing_file = path + 'test.p'
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    test_images, test_labels = test['features'], test['labels']

    for i in range(k):
        origin_class_images = test_images[test_labels == classses[i]]
        idx1 = random.sample(range(len(origin_class_images)), 10)
        confuse_class_images = test_images[test_labels == confuse_classes[i]]
        idx2 = random.sample(range(len(confuse_class_images)), 10)

        fig, axes = plt.subplots(2, 10, figsize=(12, 4))
        for j in range(10):
            axes[0, j].imshow(origin_class_images[idx1[j]])
            axes[0, j].grid(False)
            axes[0, j].axis('off')
            axes[1, j].imshow(confuse_class_images[idx2[j]])
            axes[1, j].grid(False)
            axes[1, j].axis('off')
        class_name = sign_names['SignName'].values[classses[i]]
        confuse_class_name = sign_names['SignName'].values[confuse_classes[i]]
        suptitle = class_name + '  vs  ' + confuse_class_name
        print("%s is easy to confuse with %s" % (class_name, confuse_class_name))
        fig.suptitle(suptitle, fontsize="x-large")
        plt.show()


def outputFeatureMap(image_input, activation_min=-1, activation_max=-1, plt_num=1):

    """
    image_input: the test image being fed into the network to produce the feature maps
    tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
    activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
    plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
    """

    modern_cnn_graph = tf.Graph()
    with modern_cnn_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        is_dropout = tf.placeholder(tf.bool)
        logits = get_modern_cnn(x, n_classes=n_classes, dropout=is_dropout)

        with tf.variable_scope('conv2', reuse=True):
            conv2_weights = tf.get_variable('weights')

    with tf.Session(graph=modern_cnn_graph) as sess:
        sess.run(tf.global_variables_initializer())

        tf.train.Saver().restore(sess, 'models/' + model_name)
        # If we chose to keep training previously trained model, restore session.
        print("Restore session...")

        # Here make sure to preprocess your image_input in a way your network expects
        # with size, normalization, ect if needed
        # image_input =
        # Note: x should be the same name as your network's tensorflow data placeholder variable
        # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
        # activation = conv1_weights.eval(session=sess, feed_dict={x: image_input, is_dropout: False})
        activation = sess.run(conv2_weights, feed_dict={x: image_input, is_dropout: False})
        featuremaps = activation.shape[3]
        plt.figure(plt_num, figsize=(15, 15))
        for featuremap in range(featuremaps):
            plt.subplot(8, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
            plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
            if activation_min != -1 & activation_max != -1:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                           vmax=activation_max, cmap="gray")
            elif activation_max != -1:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
            elif activation_min != -1:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
            else:
                plt.imshow(activation[:, :, 0, featuremap], interpolation="nearest", cmap="gray")


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
    X_trn_pp, y_trn_pp = preprocess_data(X_train, y_train, dataset="train", augment=1, preprocess=True)
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

    # show the augment image
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
        axes[i, 2].imshow(rotate(X_train[rnd_idx[i]]))

        axes[i, 3].grid(False)
        axes[i, 3].axis('off')
        axes[i, 3].imshow(motion_blur(X_train[rnd_idx[i]]))

        axes[i, 4].grid(False)
        axes[i, 4].axis('off')
        axes[i, 4].imshow(affine(X_train[rnd_idx[i]]))
    plt.show()

    #### Normalize the data set ####
    mean = np.mean(X_trn_pp)
    stdd = np.std(X_trn_pp)
    scaled_X_trn_pp = (X_trn_pp - mean) / stdd
    scaled_X_val_pp = (X_val_pp - mean) / stdd
    scaled_X_tst_pp = (X_tst_pp - mean) / stdd


    #### Control panel 1, setup model parameters ####
    model_name = 'new_cnn_'

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
    modern_cnn_modeling(scaled_X_trn_pp, y_trn_pp, scaled_X_val_pp, y_val_pp, scaled_X_tst_pp, y_tst_pp, params=params)

    #### Control panel 2, setup model parameters ####
    # Modify learning rate
    model_name = 'new_cnn_'

    max_epoches = 300
    batch_size = 64
    epochs_print = 5

    lr = 1e-5
    l2_lambda = 1e-3

    resume_training = True
    early_stopping_patience = 50

    params = [lr, max_epoches, batch_size, l2_lambda, model_name,
              resume_training, early_stopping_patience, epochs_print]

    ####  Fine tuning the model ###
    modern_cnn_modeling(scaled_X_trn_pp, y_trn_pp, scaled_X_val_pp, y_val_pp, scaled_X_tst_pp, y_tst_pp, params=params)



    model_name = 'new_cnn_'
    #### Evaluate the trained model ####
    evaluate_model(scaled_X_tst_pp, y_tst_pp, model_name)


    #### Analyse models ####
    # failure analyse on test dataset
    failure_analyse(scaled_X_tst_pp, y_tst_pp, model_name)

    ###  Find some images from the internet and analyse them
    new_img, new_label = read_all_imgs('input/')
    X_new = np.array(new_img)
    y_new = np.array(new_label)
    sign_names = pd.read_csv('signnames.csv')

    implot, axes = plt.subplots(2, 5, figsize=(12, 8))
    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(X_new[i])
        axes[row, col].grid(False)
        axes[row, col].axis('off')
        axes[row, col].set_title(sign_names.SignName[new_label[i]])
    plt.show()

    # preprocess the new test data
    X_new_pp = preprocess_array(X_new)

    # normalize the new test data
    scaled_X_new = (X_new_pp - mean) / stdd

    model_name = 'new_cnn_'

    #  build graph
    modern_cnn_graph = tf.Graph()
    with modern_cnn_graph.as_default():
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, n_classes)

        is_dropout = tf.placeholder(tf.bool)
        logits = get_modern_cnn(x, n_classes=n_classes, dropout=is_dropout)
        logit_probs = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # ##  evaluate models on the internet images## #
    with tf.Session(graph=modern_cnn_graph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'models/' + model_name)
        accuacy, logits_probs = sess.run([accuracy_operation, logit_probs],
                                         feed_dict={x: scaled_X_new, y: y_new, is_dropout: False})

        top5 = sess.run(tf.nn.top_k(tf.constant(logits_probs), k=5))

        for i in range(10):
            top5_classes = np.array(sign_names.SignName[top5[1][i, :]])
            top5_probs = np.array(top5[0][i, :])

            implot = plt.figure(figsize=(10, 4))

            ax = implot.add_subplot(1, 2, 1)
            ax.grid(False)
            ax.axis('off')
            ax.imshow(X_new[i])
            ax.set_title(sign_names.SignName[y_new[i]])

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

        print("=============================================")
        print("The accuracy of Internet Image = %.4f)" % accuacy)


    #### Output feature map ###
    outputFeatureMap(scaled_X_tst_pp)