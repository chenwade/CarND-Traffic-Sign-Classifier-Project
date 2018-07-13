import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, transform, img_as_ubyte, img_as_uint
import bcolz
import random
import warnings
import os

def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def YUV(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def YUV2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def YCC(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def YCC2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


def Histeq(img):
    yuv = YUV(img)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return YUV2RGB(yuv)


def Adapthisteq(img):
    return exposure.equalize_adapthist(img)


def scale(img, scale_limit=0.1):
    """
        augment data by scaling the image
    """
    scale_value = random.uniform(-scale_limit + 1, scale_limit + 1)
    height, width = img.shape[:2]
    scaled_img = cv2.resize(img, None, fx=scale_value, fy=scale_value, interpolation=cv2.INTER_AREA)
    rescaled_img = cv2.resize(scaled_img, (height, width), interpolation=cv2.INTER_AREA)
    return rescaled_img

def rotate_cv(img, rotate_limit=20):
    """
        augment data by bluring the image
    """
    height, width = img.shape[:2]
    rotate_angle = np.deg2rad(random.randint(-rotate_limit, rotate_limit))
    M = cv2.getRotationMatrix2D((height // 2, width // 2), rotate_angle, 1)
    rotated_img = cv2.warpAffine(img, M, (height, width), flags=cv2.INTER_AREA)
    return rotated_img

def rotate(img, rotate_limit=15):
    """
        augment data by bluring the image
    """
    rotate_angle = random.randint(-rotate_limit, rotate_limit)
    rotated_img = transform.rotate(img, rotate_angle, mode='edge')
    return img_as_ubyte(rotated_img)

def motion_blur(img, kernel_sz=3):
    """
        augment data by motion bluring the image
    """
    imshape = img.shape
    kernel_mb = np.zeros((kernel_sz, kernel_sz))
    kernel_mb[int((kernel_sz - 1) / 2), :] = np.ones(kernel_sz)
    kernel_mb = kernel_mb / kernel_sz
    blur = cv2.filter2D(img, -1, kernel_mb)
    return blur.reshape(*imshape)


def perturbed_cv(img, offset=2):
    """
        augment data by perturbing the image
    """
    height, width = img.shape[:2]
    h_offset = random.randint(-offset, offset)
    w_offset = random.randint(-offset, offset)
    M = np.float32([[1, 0, w_offset], [0, 1, h_offset]])
    pertubed_image = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_AREA)
    return pertubed_image

def perturbed(img, offset=2):
    """
         augment data by perturbing the image
    """

    h_offset = random.randint(-offset, offset)
    w_offset = random.randint(-offset, offset)
    tform = transform.SimilarityTransform(translation=(h_offset, w_offset))
    pertubed_image = transform.warp(img, tform, mode='edge')
    return pertubed_image


def affine(img, scale_limit=0.1, angle_limit=15., shear_limit=10., trans_limit=2):
    """
        augment data by affining the image
    """
    height, width = img.shape[:2]
    centering = np.array((height, width)) / 2. - 0.5
    # set the scale/ angle, shear, translation parameters
    scale = random.uniform(1 - scale_limit, 1 + scale_limit)
    angle = np.deg2rad(random.uniform(-angle_limit, angle_limit))
    shear = np.deg2rad(random.uniform(-shear_limit, shear_limit))
    trans_x = random.uniform(-trans_limit, trans_limit)
    trans_y = random.uniform(-trans_limit, trans_limit)

    # translate the coordinates so that the origin is at center of image
    center = transform.SimilarityTransform(translation=-centering)
    # get a transform matrix that does the scale, angle, shear, translation operation
    tform = transform.AffineTransform(scale=(scale, scale),
                                      rotation=angle,
                                      shear=shear,
                                      translation=(trans_x, trans_y))
    # translate the image to the original place
    recenter = transform.SimilarityTransform(translation=centering)
    # do the warp operation
    affine_image = transform.warp(img, (center + (tform + recenter)).inverse, mode='edge')

    return affine_image


def Augmentation(img, p_blur=0.2, p_affine=1):
    p = random.uniform(0., 1.)
    if p_affine >= p:
        img = affine(img)

    p = random.uniform(0., 1.)
    if p_blur >= p:
        img = motion_blur(img)
    return img


def img_preprocess(img):
    pp_img = gray(img)
    pp_img = Adapthisteq(pp_img).astype(np.float32)
    pp_img = pp_img.reshape(pp_img.shape + (1,))
    return pp_img


def preprocess_array(X):
    Xp = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1), dtype=np.float32)
    for i in range(X.shape[0]):
        Xp[i] = img_preprocess(X[i])
    return Xp


def augment_dataset(X, y, augs=7):
    """
        Augument the original data and then preprocess the augmented data

    :param X:
    :param y:
    :param augs:
    :return:
    """

    assert len(X) == len(y)
    # the input data should not be preprocessed, so that shape should be (None, 32, 32, 3)
    assert X.shape[3] == 3

    X_aug = np.zeros((X.shape[0] * augs, X.shape[1], X.shape[2], X.shape[3]), dtype=np.uint8)
    y_aug = np.zeros(y.shape[0] * augs)

    for i in range(X.shape[0]):
        y_aug[augs * i: augs * (i + 1)] = y[i]
        for n in range(augs):
            if n == 0:
                X_aug[i * augs] = X[i]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_aug[i * augs + n] = img_as_ubyte(Augmentation(X[i]))

    return X_aug, y_aug


def Augment_dataset1(X, y, augs=7):
    """
        Augument the preprocessed data

    :param X: 
    :param y: 
    :param augs:
    :return: 
    """

    assert len(X) == len(y)
    # the input data should be preprocessed, so that shape should be (None, 32, 32, 1)
    assert X.shape[3] == 1

    X_aug = np.zeros((X.shape[0] * augs, X.shape[1], X.shape[2], X.shape[3]), dtype=np.float32)
    y_aug = np.zeros(y.shape[0] * augs)

    for i in range(X.shape[0]):
        y_aug[augs * i: augs * (i + 1)] = y[i]
        for n in range(augs):
            if n == 0:
                X_aug[i * augs] = X[i]
            else:
                X_aug[i * augs + n] = Augmentation(X[i])

    return X_aug, y_aug



def preprocess_data(X, y, dataset, augment=7, preprocess=True):
    """
    Augument the original data and then preprocess the augmented data

     :param X:
     :param y:
     :param dataset:
     :param augment: make the dataset to its 'augment' times, if agument == 1, keep the dataset the same
     :param preprocess: whether or not use constract histogram to preprocess the data
     :return:
     """

    assert augment >= 1
    assert dataset == "train" or dataset == "valid" or dataset == "test"

    if augment == 1:
        augs_str = ''
    else:
        augs_str = '_aug' + str(augment)

    if preprocess is False:
        pp_str = ''
    else:
        pp_str = '_pp'

    X_path = 'models/X_' + dataset + augs_str + pp_str + '.dat'
    y_path = 'models/y_' + dataset + augs_str + pp_str + '.dat'

    if os.path.exists(X_path) and os.path.exists(y_path):
        X = load_array(X_path)
        y = load_array(y_path)
    else:
        if augment > 1:
            X, y = augment_dataset(X, y, augment + 1)
        if preprocess is True:
            X = preprocess_array(X)

        save_array(X_path, X)
        save_array(y_path, y)
    return X, y
