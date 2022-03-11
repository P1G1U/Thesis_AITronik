import tensorflow as tf
import pandas as pd
import numpy as np

#The raw data can easily be loaded as a Pandas DataFrame, but is not immediately usable as input to a TensorFlow model. 

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

#To build the preprocessing model, start by building a set of symbolic keras.Input objects, matching the names and data-types of the CSV columns.

inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

#The first step in your preprocessing logic is to concatenate the numeric inputs together, and run them through a normalization layer:

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
norm = tf.keras.layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

#Collect all the symbolic preprocessing results, to concatenate them later.

preprocessed_inputs = [all_numeric_inputs]

#For the string inputs use the tf.keras.layers.StringLookup function to map from strings to integer indices in a vocabulary. Next, use tf.keras.layers.CategoryEncoding to convert the indexes into float32 data appropriate for the model.
#The default settings for the tf.keras.layers.CategoryEncoding layer create a one-hot vector for each input. A layers.Embedding would also work. See the preprocessing layers guide and tutorial for more on this topic.

for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = tf.keras.layers.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

#With the collection of inputs and processed_inputs, you can concatenate all the preprocessed inputs together, and build a model that handles the preprocessing:

preprocessed_inputs_cat = tf.keras.layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

#This model just contains the input preprocessing. You can run it to see what it does to your data. Keras models don't automatically convert Pandas DataFrames because it's not clear if it should be converted to one tensor or to a dictionary of tensors. So convert it to a dictionary of tensors:

titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}

#Slice out the first training example and pass it to this preprocessing model, you see the numeric features and string one-hots all concatenated together:

features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)

#Now build the model on top of this:

def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs,result)

    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),optimizer=tf.optimizers.Adam())
    return model

titanic_model = titanic_model(titanic_preprocessing, inputs)

#When you train the model, pass the dictionary of features as x, and the label as y.

titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

#Since the preprocessing is part of the model, you can save the model and reload it somewhere else and get identical results:

titanic_model.save('test')