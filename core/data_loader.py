import numpy as np
import pandas as pd
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import os
import logging
import pdb

class DataLoader():

    def __init__(self, num_classes):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.num_samples = self.x_train.shape[0]
        self.num_classes = num_classes
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)
        self.__logger = logging.getLogger(__name__)

    def get_train_data(self):
        # normalization
        x_train = self.x_train.astype('float32')
        x_train /= 255
        x_test = self.x_test.astype('float32')
        x_test /= 255
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(self.y_train,num_classes=self.num_classes)
        y_test = keras.utils.to_categorical(self.y_test, num_classes=self.num_classes)
        self.__logger = logging.info("number of training samples: {num_train}. number of testing samples: {num_test}"
                                     .format(num_train=x_train.shape[0], num_test = x_test.shape[0]))

        return x_train, y_train, x_test, y_test
