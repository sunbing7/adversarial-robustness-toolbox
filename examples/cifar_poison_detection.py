# -*- coding: utf-8 -*-
"""Generates a backdoor for MNIST dataset, then trains a convolutional neural network on the poisoned dataset,
 and runs activation defence to find poison."""
from __future__ import absolute_import, division, print_function, unicode_literals

import pprint
import json

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence
import tensorflow

from keras.models import load_model
import os
import utils_backdoor

tensorflow.compat.v1.disable_eager_execution()

RESULT_DIR = 'cifar'
AE_TST = [440,	1061,	1258,	3826,	3942,	3987,	4831,	4875,	5024,	6445,	7133,	9609]
TARGET_LABEL = [0,0,0,0,0,0,1,0,0,0]
MODEL_ATTACKPATH = 'cifar_semantic_greencar_frog_attack.h5'
NUM_CLASSES = 10
TARGET_CLASS = 1

DATA_DIR = RESULT_DIR  # data folder
DATA_FILE = 'cifar.h5'  # dataset file

def main():
    # Read MNIST dataset (x_raw contains the original images):
    min_ = 0.
    max_ = 1.
    x_train_clean, y_train_clean, x_train_mix, y_train_mix, x_test_mix, y_test_mix, _, _, x_test_adv, y_test_adv, is_poison_train = load_dataset()

    model = load_model('%s/%s' % (RESULT_DIR, MODEL_ATTACKPATH))

    classifier = KerasClassifier(model=model, clip_values=(min_, max_))

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test_mix), axis=1)
    acc = np.sum(preds == np.argmax(y_test_mix, axis=1)) / y_test_mix.shape[0]
    print("\nTest accuracy: %.2f%%" % (acc * 100))

    # Evaluate the classifier on poisonous data
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test_adv, axis=1)) / y_test_adv.shape[0]
    print("\nPoisonous test set accuracy (i.e. effectiveness of poison): %.2f%%" % (acc * 100))

    # Evaluate the classifier on clean data
    preds = np.argmax(classifier.predict(x_train_clean), axis=1)
    acc = np.sum(preds == np.argmax(y_train_clean, axis=1)) / y_train_clean.shape[0]
    print("\nClean test set accuracy: %.2f%%" % (acc * 100))

    # Calling poisoning defence:
    defence = ActivationDefence(classifier, x_train_mix, y_train_mix)

    # End-to-end method:
    print("------------------- Results using size metric -------------------")
    print(defence.get_params())
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")

    # Evaluate method when ground truth is known:
    is_clean = is_poison_train == 0
    confusion_matrix = defence.evaluate_defence(is_clean)
    print("Evaluation defence results for size-based metric: ")
    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label])

    # Visualize clusters:
    print("Visualize clusters")
    sprites_by_class = defence.visualize_clusters(x_train_mix, RESULT_DIR + "poison_demo")
    # Show plots for clusters of class 5
    #for n_class in range (0, 10):
    n_class = TARGET_CLASS
    try:
        import matplotlib.pyplot as plt

        plt.imshow(sprites_by_class[n_class][0])
        plt.title("Class " + str(n_class) + " cluster: 0")
        #plt.show()
        plt.savefig(RESULT_DIR + "/Class " + str(n_class) + " cluster: 0" + ".png")
        plt.imshow(sprites_by_class[n_class][1])
        plt.title("Class " + str(n_class) + " cluster: 1")
        #plt.show()
        plt.savefig(RESULT_DIR + "/Class " + str(n_class) + " cluster: 1" + ".png")
    except ImportError:
        print("matplotlib not installed. For this reason, cluster visualization was not displayed")

    '''
    # Try again using distance analysis this time:
    print("------------------- Results using distance metric -------------------")
    print(defence.get_params())
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA", cluster_analysis="distance")
    confusion_matrix = defence.evaluate_defence(is_clean)
    print("Evaluation defence results for distance-based metric: ")
    jsonObject = json.loads(confusion_matrix)
    for label in jsonObject:
        print(label)
        pprint.pprint(jsonObject[label])

    # Other ways to invoke the defence:
    kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA"}
    defence.cluster_activations(**kwargs)

    kwargs = {"cluster_analysis": "distance"}
    defence.analyze_clusters(**kwargs)
    defence.evaluate_defence(is_clean)

    kwargs = {"cluster_analysis": "smaller"}
    defence.analyze_clusters(**kwargs)
    defence.evaluate_defence(is_clean)
    '''
    print("done :) ")

def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    '''
    split test set: first half for fine tuning, second half for validation
    @return
    train_clean, test_clean, train_adv, test_adv
    '''
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_test = X_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_clean = np.delete(x_test, AE_TST, axis=0)
    y_clean = np.delete(y_test, AE_TST, axis=0)

    x_adv = x_test[AE_TST]
    y_adv_c = y_test[AE_TST]
    y_adv = np.tile(TARGET_LABEL, (len(x_adv), 1))
    # randomly pick
    #'''
    idx = np.arange(len(x_clean))
    np.random.shuffle(idx)

    #print(idx)

    x_clean = x_clean[idx, :]
    y_clean = y_clean[idx, :]

    idx = np.arange(len(x_adv))
    np.random.shuffle(idx)

    #print(idx)

    DATA_SPLIT = 0.3

    x_adv = x_adv[idx, :]
    y_adv_c = y_adv_c[idx, :]
    #'''

    x_train_clean = x_clean[int(len(x_clean) * DATA_SPLIT):]
    y_train_clean = y_clean[int(len(y_clean) * DATA_SPLIT):]

    x_train_mix = np.concatenate((x_clean[int(len(x_clean) * DATA_SPLIT):], x_adv[int(len(x_adv) * DATA_SPLIT):]), axis=0)
    y_train_mix = np.concatenate((y_clean[int(len(y_clean) * DATA_SPLIT):], y_adv_c[int(len(y_adv_c) * DATA_SPLIT):]), axis=0)

    x_test_mix = np.concatenate((x_clean[:int(len(x_clean) * DATA_SPLIT)], x_adv[:int(len(x_adv) * DATA_SPLIT)]), axis=0)
    y_test_mix = np.concatenate((y_clean[:int(len(y_clean) * DATA_SPLIT)], y_adv_c[:int(len(y_adv_c) * DATA_SPLIT)]), axis=0)

    # is poison in x_train_mix
    is_poison_train = np.append(np.zeros(len(x_clean[int(len(x_clean) * DATA_SPLIT):])), np.ones(len(x_adv[int(len(x_adv) * DATA_SPLIT):])), axis=0)

    x_train_adv = x_adv[int(len(y_adv) * DATA_SPLIT):]
    y_train_adv = y_adv[int(len(y_adv) * DATA_SPLIT):]
    x_test_adv = x_adv[:int(len(y_adv) * DATA_SPLIT)]
    y_test_adv = y_adv[:int(len(y_adv) * DATA_SPLIT)]

    return x_train_clean, y_train_clean, x_train_mix, y_train_mix, x_test_mix, y_test_mix, x_train_adv, y_train_adv, x_test_adv, y_test_adv, is_poison_train


if __name__ == "__main__":
    main()
