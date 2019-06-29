#!/usr/bin/env python3
"""
Model evaluation script using the test dataset.

@author: Netanel Azoulay
@author: Roman Koifman
"""

from keras.preprocessing.image import ImageDataGenerator
from projectParams import *
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Evaluation params
size = nbatch  # Maximum Test Batch size.


print("\nLoading Model and Weights..")
model = load_model(modelPath)
model.load_weights(modelWeights)

# Test Data (20%).
print("Loading test data..")
test_datagen = ImageDataGenerator(rescale=1. / 255.,
                                  rotation_range=10,  # randomly rotate up to 40 degrees.
                                  width_shift_range=0.2,  # randomly shift range.
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  fill_mode="nearest")  # fill new pixels created by shift

test_generator = test_datagen.flow_from_directory(testFolder,
                                                  target_size=(imgDim, imgDim),
                                                  color_mode='grayscale',
                                                  batch_size=nbatch,
                                                  classes=classes,
                                                  class_mode="categorical")
count = 1
x_test, y_test = [], []
for ibatch, (x, y) in enumerate(test_generator):
    sys.stdout.write("\rBatch Progress: %d%%" % (count * 100 / size))
    sys.stdout.flush()
    count += 1
    x_test.append(x)
    y_test.append(y)
    ibatch += 1
    if ibatch == size:
        break

# Concatenate everything together
x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)
y_test = np.int32([np.argmax(r) for r in y_test])

# Get the predictions from the model and calculate the accuracy
print("\nPredicting..")
y_pred = np.int32([np.argmax(r) for r in model.predict(x_test, verbose=1)])

match = (y_test == y_pred)

sumString = 'Testing Accuracy = %.2f%%' % (np.sum(match) * 100 / match.shape[0])
print(sumString)

try:
    text_file = open(logFolder + "/TestAccuracy.txt", "w")
    text_file.write(sumString)
    text_file.close()
except OSError:
    pass

# Confusion matrix

plt.figure(figsize=(9, 8))
cm = confusion_matrix(y_test, y_pred)
cm = cm / cm.sum(axis=1)
sn.heatmap(cm, annot=True)
plt.savefig(logFolder + '/confusionMat.png', bbox_inches='tight')
