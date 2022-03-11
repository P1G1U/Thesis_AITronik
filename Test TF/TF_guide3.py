import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

#For any small CSV dataset the simplest way to train a TensorFlow model on it is to load it into memory as a pandas Dataframe or a NumPy array.

#A relatively simple example is the abalone dataset.
#    The dataset is small.
#    All the input features are all limited-range floating point values.

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

#The nominal task for this dataset is to predict the age from the other measurements, so separate the features and labels for training:

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age').values
#For this dataset you will treat all features identically. Pack the features into a single NumPy array.:

abalone_features = np.array(abalone_features)

#Next make a regression model predict the age. Since there is only a single input tensor, a keras.Sequential model is sufficient here.

abalone_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])

abalone_model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

#To train that model, pass the features and labels to Model.fit:

abalone_model.fit(abalone_features, abalone_labels, epochs=8)

#You have just seen the most basic way to train a model using CSV data. Next, you will learn how to apply preprocessing to normalize numeric columns.

#It's good practice to normalize the inputs to your model. The Keras preprocessing layers provide a convenient way to build this normalization into your model.
#The layer will precompute the mean and variance of each column, and use these to normalize the data.

#First you create the layer:

normalize = tf.keras.layers.Normalization()

#Then you use the Normalization.adapt() method to adapt the normalization layer to your data.

normalize.adapt(abalone_features)

#Then use the normalization layer in your model:

norm_abalone_model = tf.keras.Sequential([
    normalize,
    abalone_model
])

norm_abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                           optimizer = tf.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=8)

