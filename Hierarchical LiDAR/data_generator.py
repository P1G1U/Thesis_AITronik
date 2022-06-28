import tensorflow as tf
import numpy as np
from data_utils import pnt_on_grid, quant_rot
from data_mgmt import DAO

class CustomDataGen(tf.keras.utils.Sequence):
  
    def __init__(self, df_features, df_targets,
                 map_img,
                 pixel_scale, 
                 batch_size,
                 input_size=(100,100,1),
                 shuffle=False):
        
        self.df_features = df_features.copy()
        self.df_targets = df_targets.copy()
        self.map = map_img
        self.pixel_scale = pixel_scale
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = df_features.shape[0]
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_output_pos(self, batches):
        pos_masks = []

        for i in batches.numpy():
            map_grid_inst = pnt_on_grid(self.map, i,7,self.pixel_scale)
            pos_masks.append(map_grid_inst)
        
        return pos_masks

    def __get_output_rot(self, batches):
        rot_masks = []

        for i in batches.numpy():
            rot_grid_inst = quant_rot(i,90, 2)
            rot_masks.append(rot_grid_inst)

        return rot_masks

    def __get_data(self, batches_features, batches_targets):
        # Generates data containing batch_size samples

        #map_array = []

        #for i in range(batches_features.shape[0]):
        #    map_array.append(self.map)

        batches_targets_pos=tf.gather(batches_targets,[0,1],axis=1)
        batches_targets_rot=tf.gather(batches_targets,[2],axis=1)

        y_pos_batch = self.__get_output_pos(batches_targets_pos)
        y_rot_batch = self.__get_output_rot(batches_targets_rot)

        #map_array = tf.reshape(map_array,[self.batch_size,self.map.shape[0],self.map.shape[1],self.map.shape[2]])
        batches_features = tf.reshape(batches_features,[batches_features.shape[0],batches_features.shape[1],1])

        y_pos_batch = tf.reshape(y_pos_batch,[self.batch_size,y_pos_batch[0].shape[0],y_pos_batch[0].shape[1],y_pos_batch[0].shape[2]])
        y_rot_batch = tf.reshape(y_rot_batch,[self.batch_size,y_rot_batch[0].shape[0],1])

        return batches_features, [y_pos_batch, y_rot_batch]

    def __getitem__(self, index):
        
        batches_features = self.df_features[index * self.batch_size:(index + 1) * self.batch_size]
        batches_targets = self.df_targets[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches_features, batches_targets)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

def initialize_gen(dao, map, pixel_scale, batch_size, input_size):

    dao.read()
    dao.divide_data()

    TR_gendata = CustomDataGen(dao.TR_features, dao.TR_targets, map, pixel_scale, batch_size, (map.shape[0],map.shape[1],map.shape[2]))
    TS_gendata = CustomDataGen(dao.TS_features, dao.TS_targets, map, pixel_scale, batch_size, (map.shape[0],map.shape[1],map.shape[2]))

    return TR_gendata, TS_gendata
