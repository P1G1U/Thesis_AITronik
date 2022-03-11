#Import TensorFlow into your program to get started:

import tensorflow as tf

#Load and prepare the MNIST dataset. Convert the sample data from integers to floating-point numbers:

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build a tf.keras.Sequential model by stacking layers.

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#Define a loss function for training using losses.SparseCategoricalCrossentropy, which takes a vector of logits and a True index and returns a scalar loss for each example.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#Before you start training, configure and compile the model using Keras Model.compile. Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

#Use the Model.fit method to adjust your model parameters and minimize the loss: 

model.fit(x_train, y_train, epochs=5)

#The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".

model.evaluate(x_test, y_test, verbose=2)