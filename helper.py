import numpy as np
"""
Forked from:
https://github.com/navoshta/traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb
"""
import time
import matplotlib.pyplot as plt
import cv2
import os
import pickle

class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(self, saver, session, model_name, patience=100, minimize=True):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     :
                    TensorFlow Saver object to be used for saving and restoring model.
        session   :
                    TensorFlow Session object containing graph where model is restored.
        patience  :
                    Early stopping patience. This is the number of epochs we wait for
                    accuracy to start improving again before stopping and restoring
                    previous best performing parameters.

        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = None
        self.model_name = model_name

    def __call__(self, value, epoch):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
        value     :
                    Last epoch monitored value.
        epoch     :
                    Last epoch number.

        Returns
        -------
        `True` if we waited enough and it's time to stop and we restored the
        best performing weights, or `False` otherwise.
        """
        if (self.minimize and value < self.best_monitored_value) or (
                not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.restore_path = self.saver.save(self.session,  "models/" + self.model_name)
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True

        return False



def load_traffic_sign_data(path):
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

    assert (len(X_train) == len(y_train))
    assert (len(X_valid) == len(y_valid))
    assert (len(X_test) == len(y_test))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_time_hhmmss(start = None):
    """
    Calculates time since `start` and formats as a string.
    """
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str


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
    labels = []

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
            label = file.split('.')[0]
            labels.append(int(label))

    return images, labels