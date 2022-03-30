import model as m
import numpy as np
import os

os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=256)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model= m.NN_Model()
model.model_define(0.0)

model.model_cnn.load_weights("models/cp-epoch-train.ckpt")

def get_data(filename):
    data_test = m.DAO(filename)

    data_test.read()
    #data_test.divide_data(0.25)

    data_tensor_test=m.DAO()

    data_tensor_test.features=tf.reshape(data_test.features,[-1,720,2,1])
    data_tensor_test.targets=tf.reshape(data_test.targets,[-1,3])

    return data_test, data_tensor_test

def evaluate(model, data_test, data_tensor_test):
    result = model.model_cnn.predict(data_tensor_test.features)
    pos_test = result[:,0:2]
    rot_test = result[:,2]
    pos_real = np.array([data_test.targets["pos_x"].to_list(),data_test.targets["pos_y"].to_list()]).T
    rot_real = (data_test.targets["pos_yaw"]).to_numpy()

    return pos_test, rot_test, pos_real, rot_real

print("loading 2203 ....")

data, data_tensor = get_data("laser_log/laser_log2203.csv")

print("evaluate 2203 ....")

pos_test, rot_test, pos_real, rot_real = evaluate(model, data, data_tensor)

with open('predict_data2203.csv', 'w') as f:
    for i in range(len(rot_real)):
        f.write("{};{};{};{};{};{}\n".format(pos_real[i][0],pos_real[i][1],rot_real[i],pos_test[i][0],pos_test[i][1],rot_test[i]))

print("loading 2203 with walls ....")

data, data_tensor = get_data("laser_log/laser_log2203w.csv")

print("evaluating 2203 with walls ....")

pos_test, rot_test, pos_real, rot_real = evaluate(model, data, data_tensor)

with open('predict_data2203w.csv', 'w') as f:
    for i in range(len(rot_real)):
        f.write("{};{};{};{};{};{}\n".format(pos_real[i][0],pos_real[i][1],rot_real[i],pos_test[i][0],pos_test[i][1],rot_test[i]))

print("loading 2303 ....")

data, data_tensor = get_data("laser_log/laser_log2303.csv")

print("evaluating 2303 ....")

pos_test, rot_test, pos_real, rot_real = evaluate(model, data, data_tensor)

with open('predict_data2303.csv', 'w') as f:
    for i in range(len(rot_real)):
        f.write("{};{};{};{};{};{}\n".format(pos_real[i][0],pos_real[i][1],rot_real[i],pos_test[i][0],pos_test[i][1],rot_test[i]))