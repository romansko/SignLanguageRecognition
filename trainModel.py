"""
Inspired by https://github.com/jaredvasquez/CNN-HowManyFingers/blob/master/trainModel.ipynb
"""

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from projectParams import classes, modelWeights, modelPath, logFolder
from shutil import copyfile
from cnn12 import imgDim, getModel
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

trainWeights = 'trainWeights.h5'  # weights to save
epochs = 50

# HyperParams
nbatch = 128  # 32 default.


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            copyfile(trainWeights, "Temp/epoch" + str(epoch) + "_weights.h5")
        except OSError:
            pass
        return


def trainModel():
    # ImageDataGenerator purpose:
    # Label the data from the directories.
    # Augment the data with shifts, rotations, zooms, and mirroring.
    # Mirroring will help to ensure that the data are not biased to a particular handedness.
    # Changes are applied Randomly.

    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=10,  # randomly rotate up to 40 degrees.
                                       width_shift_range=0.2,  # randomly shift range.
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode="nearest")  # fill new pixels created by shift

    train_generator = train_datagen.flow_from_directory('images/train/',
                                                        target_size=(imgDim, imgDim),
                                                        color_mode='grayscale',
                                                        batch_size=nbatch,
                                                        classes=classes,
                                                        class_mode="categorical")

    # Validation Data (10% of train data)

    valid_datagen = ImageDataGenerator(rescale=1. / 255.)

    valid_generator = valid_datagen.flow_from_directory('images/validation/',
                                                        target_size=(imgDim, imgDim),
                                                        color_mode='grayscale',
                                                        batch_size=nbatch,
                                                        classes=classes,
                                                        class_mode="categorical")

    model = getModel(weightsPath=modelWeights)
    model.save(modelPath)
    model.summary()

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_test = valid_generator.n // valid_generator.batch_size

    csv_logger = CSVLogger(logFolder + '/training.csv')

    # EarlyStopping = method to stop training when a monitored quantity has stopped improving.
    # Define a callback.Set monitor as val_acc, patience as 5 and mode as max so that if val_acc does not improve over 5
    # epochs, terminate the training process.
    # can also do on val_loss.
    ccb = CustomCallback()
    callbacks_list = [
        # EarlyStopping(monitor='val_acc', patience=5, mode='max'),
        ModelCheckpoint(filepath=trainWeights, monitor='val_acc'),  # save_best_only=True),
        ccb,
        csv_logger
    ]

    # os.environ["CUDA_VISIBLE_DEVICES"]="0"      # visible devices
    # import tensorflow as tf
    # with tf.device('cpu:0'):                    # Device to run on. cpu:0 / gpu:0 / gpu:1.
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
