#!/usr/bin/env python3
"""
Script for training the cnn12 model.
Based on https://github.com/jaredvasquez/CNN-HowManyFingers/blob/master/trainModel.ipynb

@author: Netanel Azoulay
@author: Roman Koifman
"""

import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from projectParams import *
from shutil import copyfile
from cnn12 import imgDim, getModel
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Training Params.
trainWeights = 'trainWeights.h5'  # weights to save
epochs = 50


class CustomCallback(keras.callbacks.Callback):
    """
    Custom callback class in order to save weights after each epoch.
    Was used to backup weights via Azure Virtual Machine.
    """

    def on_epoch_end(self, epoch, logs=None):
        try:
            copyfile(trainWeights, "Temp/epoch" + str(epoch) + "_weights.h5")
        except OSError:
            pass
        return


def trainModel():
    """
    Train the CNN12 model by Loading Training and Validation data.
    At the end of the training a learning graph will be plotted.
    """

    # Load training data with augmentation.
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=10,  # randomly rotate up to 40 degrees.
                                       width_shift_range=0.2,  # randomly shift range.
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode="nearest")  # fill new pixels created by shift

    train_generator = train_datagen.flow_from_directory(trainFolder,
                                                        target_size=(imgDim, imgDim),
                                                        color_mode='grayscale',
                                                        batch_size=nbatch,
                                                        classes=classes,
                                                        class_mode="categorical")

    # # Load validation data (10% of original train data).

    valid_datagen = ImageDataGenerator(rescale=1. / 255.)

    valid_generator = valid_datagen.flow_from_directory(validFolder,
                                                        target_size=(imgDim, imgDim),
                                                        color_mode='grayscale',
                                                        batch_size=nbatch,
                                                        classes=classes,
                                                        class_mode="categorical")

    model = getModel(weightsPath=modelWeights)  # Build cnn12 model.
    model.save(modelPath)
    model.summary()

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_test = valid_generator.n // valid_generator.batch_size

    csv_logger = CSVLogger(logFolder + '/training.csv')

    ccb = CustomCallback()
    callbacks_list = [
        ModelCheckpoint(filepath=trainWeights, monitor='val_acc'),
        ccb,
        csv_logger
    ]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=step_size_train,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=step_size_test,
        callbacks=callbacks_list)

    # save weights
    model.save_weights(trainWeights)

    # Plot train graphs
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    nepochs = len(history.history['loss'])
    plt.plot(range(nepochs), history.history['loss'], 'r-', label='train')
    plt.plot(range(nepochs), history.history['val_loss'], 'b-', label='validation')
    plt.legend(prop={'size': epochs})
    plt.ylabel('loss')
    plt.xlabel('# of epochs')
    plt.subplot(1, 2, 2)
    plt.plot(range(nepochs), history.history['acc'], 'r-', label='train')
    plt.plot(range(nepochs), history.history['val_acc'], 'b-', label='validation')
    plt.legend(prop={'size': epochs})
    plt.ylabel('accuracy')
    plt.xlabel('# of epochs')
    plt.savefig(logFolder + '/graph.png')


if __name__ == '__main__':
    trainModel()
