# Thesis_AITronik
Global localization of an autonomous vehicle using Deep Neural Network

In the folder 'test' you can found some guide to take confidence with the Tensorflow library, taken from the tutorials of its site (https://www.tensorflow.org/tutorials).

The external library required to run those experiments are tensorflow(2.8), numpy(1.22.3) and pandas(1.4.1). The version of python used is the 3.9.10.

The next folders are the ones where you can find the models with everything you need to train and test them. 
All the next file are in the folders: the file data_mgmt.py is the data access object of the model that interact with the input file in order to get and format the data for the model, the define_model.ipynb is a python notebook where the model is created and tested, the model.py is the file that store the class of the model with all the parameters, then script_gridsearch, script_model,tune_model_param.py and valuation.py are file for the testing and tuning of the model.
There are also folders that regroup all the datasheets useful: data/ have the inputs, models/ save the model weigths and metrics/ have the results of the model 

In folder 'Basis' you can find the basic model with LiDAR instance as input and position and rotation (for unidir only position) as output

In folder 'LiDAR Map' you can find the evolution of the model with LiDAR instance and the map as input and 2 version of output: the first is as before, the second has the position and the rotation distretized. There are files for the manipulation of the data and the creation of aritifial LiDAR like create_map_db.py, preproc_data.py and process_map.py