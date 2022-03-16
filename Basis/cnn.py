"""Simple convolutional neural network"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers, models
from data_mgmt import DAO

def model_define(features, drop_rate):

    tf.summary.image("laser_inst", features)

    model = models.Sequential()

    model.add(layers.Conv2D(
        filters=32, kernel_size=3, padding="same",
        name="conv_1" ))
    model.add(layers.Conv2D(
        filters=64, kernel_size=3, padding="same",
        name="conv_2" ))
    model.add(layers.MaxPooling2D(
        pool_size=2, strides=2, padding="same",
        name="pool_1" ))
    model.add(layers.Conv2D(
        filters=256, kernel_size=3, padding="same",
        name="conv_3" ))
    model.add(layers.MaxPooling2D(
        pool_size=2, strides=2, padding="same",
        name="pool_2" ))

    model.add(layers.Flatten())

    model.add(layers.Dropout(drop_rate))
    model.add(layers.Dense(1024, name="dense_1"))

    model.add(layers.Dense(128, activation="relu", name="dense_2"))

    model.add(layers.Dense(3, activation="relu", name="dense_3"))

    return model

filename="laser_log/laser_log.csv"

print("Read data from {} .....".format(filename))

data = DAO(filename)

data.read()
data.divide_data(0.25)

print("End Reading")
print("Create the model .....")

model_cnn = model_define(data.features, 0.2)

print("End Creation")
print("Fit the model to the data .....")

model_cnn.compile(optimizer = "adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#history = model_cnn.fit(data.TR_features, data.TR_targets, epochs=10, validation_data=(data.TS_features, data.TS_targets))

print("End program.")