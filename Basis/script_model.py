"""file used by script_gridsearch to launch the gridsearch of the models"""

import model as m
import numpy as np
import sys

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

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model= m.NN_Model()
model.get_data("laser_log/datasetwall.csv")

beta=float(sys.argv[1])
drop=float(sys.argv[2])
epochs=100
batch_size=16

group_epochs=1

results=[]

model.model_define(drop_rate=drop)
model.set_beta(beta)
model.model_compile("adam",model.pos_loss,['mean_absolute_error'])
for step in range(epochs//group_epochs):
    history = model.model_run(group_epochs,batch_size, verbose=1)
    pos_mse,rot_mse,pos_mae,rot_mae = model.get_error(verbose=1)
    results.append({"beta":beta,"drop-out":drop,"epochs":(step+1)*group_epochs,"history":history,"position squared error":pos_mse, "rotation squared error":rot_mse, "position absolute error":pos_mae, "rotation absolute error":rot_mae})  

with open(sys.argv[3], 'w') as f:
    #f.write("beta;dropout;epochs;pos_mse;pos_mae;rot_mse;rot_mae\n")
    for item in results:
        f.write("{};{};{};{};{}\n".format(item["beta"],item["drop-out"],item["epochs"],item["position absolute error"],item["rotation absolute error"]))