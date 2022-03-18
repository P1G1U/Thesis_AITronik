import tensorflow as tf
from tensorflow.keras import layers, models, backend
from data_mgmt import DAO
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class NN_Model:

    model_cnn=None
    data=None
    data_tensor=None
    beta_loss = 0

    def __init__(self):
        pass

    def model_define(self, drop_rate):

        model = models.Sequential()

        model.add(layers.Conv2D(
            32, (11,1), padding="same", input_shape=(720,1,1),
            name="conv_1" ))
        model.add(layers.Conv2D(
            64, kernel_size=(5,1), padding="same",
            name="conv_2" ))
        model.add(layers.MaxPooling2D(
            (2,1), strides=2, padding="same",
            name="pool_1" ))
        model.add(layers.Conv2D(
            256, kernel_size=(5,1), padding="same",
            name="conv_3" ))
        model.add(layers.MaxPooling2D(
            (2,1), strides=4, padding="same",
            name="pool_2" ))

        model.add(layers.Flatten())

        model.add(layers.Dropout(drop_rate))
        model.add(layers.Dense(1024, activation="relu", name="dense_1"))

        model.add(layers.Dense(128, activation="relu", name="dense_2"))

        model.add(layers.Dense(3, name="dense_3"))

        self.model_cnn = model   

    def get_data(self, filename):
        self.data = DAO(filename)
        self.data.read()
        self.data.divide_data(0.25)

        #brutto ma funziona
        self.data_tensor=DAO(filename)
        self.data_tensor.read()
        self.data_tensor.divide_data(0.25)

        self.data_tensor.TR_features=tf.reshape(self.data_tensor.TR_features,[-1,720,1,1])
        self.data_tensor.TS_features=tf.reshape(self.data_tensor.TS_features,[-1,720,1,1])
        self.data_tensor.TR_targets=tf.reshape(self.data_tensor.TR_targets,[-1,3])
        self.data_tensor.TS_targets=tf.reshape(self.data_tensor.TS_targets,[-1,3])

    def model_compile(self, optimizer, loss, metrics):
        self.model_cnn.compile(optimizer=optimizer,
            loss=loss,
            metrics=metrics)
    
    def model_run(self, epochs, batch_size=32, verbose=0):
        return self.model_cnn.fit(self.data_tensor.TR_features, 
            self.data_tensor.TR_targets, 
            epochs=epochs, 
            validation_data=(self.data_tensor.TS_features, self.data_tensor.TS_targets),
            verbose=verbose,
            batch_size=batch_size
            )

    def get_error(self, verbose=0):
        result = self.model_cnn.predict(self.data_tensor.TS_features)
        pos_test = result[:,0:2]
        rot_test = result[:,2]

        pos_real = np.array([self.data.TS_targets["pos_x"].to_list(),self.data.TS_targets["pos_y"].to_list()]).T
        rot_real = (self.data.TS_targets["pos_yaw"]).to_numpy()

        pos_mse = mean_squared_error(pos_real,pos_test)
        rot_mse = mean_squared_error(rot_real,rot_test)

        pos_mae = mean_absolute_error(pos_real,pos_test)
        rot_mae = mean_absolute_error(rot_real,rot_test)

        if verbose==1:
            print("position mean square error: {} -- rotation mean square error: {} -- position mean absolute error: {} -- rotation mean absolute error: {}".format(pos_mse,rot_mse,pos_mae,rot_mae))

        return (pos_mse, rot_mse, pos_mae, rot_mae)

    def set_beta(self, value):
        self.beta_loss=value

    def pos_loss(self,y_actual, y_pred):
        loss_value = (1-self.beta_loss)*(backend.sqrt(backend.pow((y_actual[:,0]-y_pred[:,0]),2)+backend.pow((y_actual[:,1]-y_pred[:,1]),2)))+(self.beta_loss*(backend.sqrt(backend.pow((y_actual[:,2]-y_pred[:,2]),2))))
        return loss_value
