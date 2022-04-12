import tensorflow as tf
from tensorflow.keras import layers, models, losses
from keras import backend
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

    def model_define_1(self, drop_rate):

        self.num_model = 1

        backend.clear_session()

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
            (2,1), strides=2, padding="same",
            name="pool_2" ))

        model.add(layers.Flatten())

        model.add(layers.Dropout(drop_rate))
        model.add(layers.Dense(1024, activation="relu", name="dense_1"))

        model.add(layers.Dropout(drop_rate))
        model.add(layers.Dense(128, activation="relu", name="dense_2"))

        #model.add(layers.Dense(3, name="dense_3"))
        model.add(layers.Dense(2, name="dense_3"))

        self.model_cnn = model   

    def model_define_2(self, drop_rate):
        
        self.num_model = 2

        inputs = layers.Input(shape=(720,1,1))

        conv_1 = layers.Conv2D(
            16, (11,1), padding="same",
            name="conv_1" )(inputs)
        conv_2 = layers.Conv2D(
            32, kernel_size=(5,1), padding="same",
            name="conv_2" )(conv_1)
        pool_1 = layers.MaxPooling2D(
            (2,1), strides=2, padding="same",
            name="pool_1" )(conv_2)
        conv_3 = layers.Conv2D(
            128, kernel_size=(5,1), padding="same",
            name="conv_3" )(pool_1)
        pool_2 = layers.MaxPooling2D(
            (2,1), strides=2, padding="same",
            name="pool_2" )(conv_3)

        flat = layers.Flatten()(pool_2)
        flat_dropout = layers.Dropout(drop_rate)(flat)
        
        #position
        pos_dense_1 = layers.Dense(512, activation="relu", name="hidden_pos_1")(flat_dropout)
        pos_dropout = layers.Dropout(drop_rate)(pos_dense_1)
        pos_dense_2 = layers.Dense(128,name="hidden_pos_2")(pos_dropout)
        pos_result = layers.Dense(2,name="pos_output")(pos_dense_2)

        #rotation
        rot_dense_1 = layers.Dense(512, activation="relu", name="hidden_rot_1")(flat_dropout)
        rot_dropout = layers.Dropout(drop_rate)(rot_dense_1)
        rot_dense_2 = layers.Dense(128,name="hidden_rot")(rot_dropout)
        rot_result = layers.Dense(1,name="rot_output")(rot_dense_2)

        self.model_cnn = models.Model(inputs, [pos_result,rot_result])

    def get_data(self, filename):
        self.data = DAO(filename)
        self.data.read()
        self.data.divide_data(0.25)
        
        self.data_tensor=DAO()

        self.data_tensor.TR_features=tf.reshape(self.data.TR_features,[-1,720,1,1])
        self.data_tensor.TS_features=tf.reshape(self.data.TS_features,[-1,720,1,1])
        self.data_tensor.TR_targets = tf.reshape(self.data.TR_targets,[-1,3])
        self.data_tensor.TS_targets = tf.reshape(self.data.TS_targets,[-1,3])
        
        if self.num_model == 1:
            self.data_tensor.TR_targets = tf.gather(self.data_tensor.TR_targets,[0,1],axis=1)
            self.data_tensor.TS_targets = tf.gather(self.data_tensor.TS_targets,[0,1],axis=1)

        if self.num_model == 2:
            self.data_tensor.TR_targets_pos = tf.gather(self.data_tensor.TR_targets,[0,1],axis=1)
            self.data_tensor.TS_targets_pos = tf.gather(self.data_tensor.TS_targets,[0,1],axis=1)
            self.data_tensor.TR_targets_rot = tf.gather(self.data_tensor.TR_targets,[2],axis=1)
            self.data_tensor.TS_targets_rot = tf.gather(self.data_tensor.TS_targets,[2],axis=1)

    def model_compile(self, optimizer, loss, metrics):
        self.model_cnn.compile(optimizer=optimizer,
            loss=loss,
            metrics=metrics)
    
    def model_run(self, epochs, batch_size=32, verbose=0):
        if self.num_model == 1:
            return self.model_cnn.fit(self.data_tensor.TR_features, 
                self.data_tensor.TR_targets,
                epochs=epochs, 
                validation_data=(self.data_tensor.TS_features, self.data_tensor.TS_targets),
                verbose=verbose,
                batch_size=batch_size
                )
        if self.num_model == 2:
            return self.model_cnn.fit(self.data_tensor.TR_features, 
                [self.data_tensor.TR_targets_pos,self.data_tensor.TR_targets_rot], 
                epochs=epochs, 
                batch_size=batch_size,
                verbose=verbose,
                validation_data=(self.data_tensor.TS_features, [self.data_tensor.TS_targets_pos,self.data_tensor.TS_targets_rot])
                )
        else: 
            return None

    def get_error(self, verbose=0):
        
        if self.num_model == 1:
            result = self.model_cnn.predict(self.data_tensor.TS_features)

            pos_test = result[:,0:2]
            pos_real = np.array([self.data.TS_targets["pos_x"].to_list(),self.data.TS_targets["pos_y"].to_list()]).T
            pos_mse = mean_squared_error(pos_real,pos_test)
            pos_mae = mean_absolute_error(pos_real,pos_test)

        if self.num_model == 2:
            result = self.model_cnn.predict(self.data_tensor.TS_features)

            pos_test = result[0]
            pos_real = np.array([self.data.TS_targets["pos_x"].to_list(),self.data.TS_targets["pos_y"].to_list()]).T
            pos_mse = mean_squared_error(pos_real,pos_test)
            pos_mae = mean_absolute_error(pos_real,pos_test)

            rot_test = result[1]
            rot_real = (self.data.TS_targets["pos_yaw"]).to_numpy()
            rot_mse = mean_squared_error(rot_real,rot_test)
            rot_mae = mean_absolute_error(rot_real,rot_test)

        if verbose==1:
            if self.num_model == 1:
                print("position mean square error: {} -- position mean absolute error: {} ".format(pos_mse,pos_mae))
            if self.num_model == 2:
                print("position mean square error: {} -- rotation mean square error: {} -- position mean absolute error: {} -- rotation mean absolute error: {}".format(pos_mse,rot_mse,pos_mae,rot_mae))

        if self.num_model == 1:
            return (pos_mse, pos_mae)
        if self.num_model == 2:
            return (pos_mse, rot_mse, pos_mae, rot_mae)
        else: 
            return None


    def set_beta(self, value):
        self.beta_loss=value

    def pos_loss_1(self,y_actual, y_pred):
        loss_value = backend.sqrt(backend.pow((y_actual[:,0]-y_pred[:,0]),2)+backend.pow((y_actual[:,1]-y_pred[:,1]),2))
        
        return loss_value
    
    def pos_loss_2(self,y_true, y_pred):
        pos_loss = losses.MeanSquaredError()(y_true[0], y_pred[0])
        rot_loss = losses.MeanAbsoluteError()(y_true[1], y_pred[1])

        return layers.Add()([pos_loss, rot_loss])
