"""
A deep network was constructed accepting input of image dimensions of 128 x 128 with one channel (grayscale images).
The network was constructed by following Karpathy's tutorial.
Based on https://github.com/hemavakade/CNN-for-Image-Classification
"""

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.models import Sequential
from keras.optimizers import adam
from projectParams import classes, imgDim, learnRate


def getModel(weightsPath=None):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(imgDim, imgDim, 1)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(32, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(64, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(128, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(256, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(512, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(1028, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())

    model.add(Conv2D(1028, (3, 3)))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # Dense = Fully connected layer
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(classes), activation='softmax'))

    # opt = SGD(lr=learnRate, decay=1e-6, momentum=0.9, nesterov=True)
    opt = adam(lr=learnRate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    if weightsPath:
        try:
            model.load_weights(weightsPath)
            print("cnn12 weights loaded.")
        except OSError:
            print("Failed loading cnn12 weights!")
    else:
        print("cnn12 weights are not provided.")

    return model
