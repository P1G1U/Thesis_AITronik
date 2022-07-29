from skimage import io
from skimage.transform import resize
from data_utils import get_pos_from_list_fp, list_from_fingerprint, list_from_rotation, get_rot_from_slice, zoom_in
from model_utils import model_1_predict, model_2_predict
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

original_map = io.imread("data/{}".format(sys.argv[1]))

original_map = original_map[:,:,0]
original_map = original_map / 255

with open("data/laser/{}".format(sys.argv[2]), "r") as data:
    laser_inst = pd.read_csv(data, delimiter=";").copy()

max_pixel = 10000
pixel_scale = 0.1
tot_pixel = original_map.shape[0]*original_map.shape[1]
batch_size = 1024

laser_inst = tf.reshape(laser_inst, [360,1])

if tot_pixel > max_pixel:
    rate_rescale = np.sqrt(max_pixel/tot_pixel) 

else:
    rate_rescale = 1 

pixel_scale = pixel_scale/rate_rescale
new_height = int(original_map.shape[0] * rate_rescale)
new_width = int(original_map.shape[1] * rate_rescale)
map_img = resize(original_map,(new_height,new_width))

map_img_tensor = tf.reshape(map_img, [map_img.shape[0],map_img.shape[1],1])

#phase 1

phase_1_map, phase_1_rot = model_1_predict(map_img_tensor, laser_inst)

#elaborate data from phase 1

phase_1_pos = get_pos_from_list_fp(list_from_fingerprint(phase_1_map[0],pixel_scale))
phase_1_rot = np.abs(get_rot_from_slice(list_from_rotation(phase_1_rot[0]),90))

x_real = phase_1_pos[0]
y_real = phase_1_pos[1]

dim_zoom = 0.1#np.sqrt(rate_rescale)
pixel_zoom = 0.1/dim_zoom

sub_map, x_0, y_0 = zoom_in(original_map, [y_real,x_real], new_width, new_height, dim_zoom)

#phase 2

sub_map = tf.reshape(sub_map, [sub_map.shape[0],sub_map.shape[1],1])

phase_2_pos = model_2_predict(sub_map, laser_inst)

#elaborate data from phase 2

phase_2_pos = tf.reshape(phase_2_pos,[phase_2_pos.shape[1],phase_2_pos.shape[2],2])

list_fp_pred = list_from_fingerprint(phase_2_pos,pixel_zoom)

if len(list_fp_pred) > 0:

    centroid_pred = get_pos_from_list_fp(list_fp_pred)

    centroid_pred[0] += x_0*pixel_zoom
    centroid_pred[1] += y_0*pixel_zoom

else:

    print("Model 2 not used")
    centroid_pred = phase_1_pos

with open('metrics/result_path.csv', 'a+') as f:
    f.write("{};{};{};{};{}\n".format(x_real, y_real, centroid_pred[0], centroid_pred[1], phase_1_rot))


#print("phase 1 position: [{};{}]".format(x_real, y_real))

#print("position: [{};{}] rotation: {}".format(centroid_pred[0],centroid_pred[1],np.radians(phase_1_rot)))