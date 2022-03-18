import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np

class DAO:

    features = None
    targets = None

    TR_features = None
    TS_features = None
    TR_targets = None
    TS_targets = None

    add_time = False

    def __init__(self, filename, time=False):
        self.filename = filename
        self.add_time = time

    def read(self):
        
        with open("data/"+self.filename, "r") as data:
            laser_db = pd.read_csv(data, delimiter=";").copy()

        laser_list= pd.DataFrame()
        target_list= pd.DataFrame()
        inst = 1 #il counter delle istanze inizia da 1
        #laser_angle=[]
        #laser_range=[]
        laser_inst=pd.Series(dtype="float32")
        target_inst=pd.Series(dtype="float32")

        for i in laser_db.values:
            if i[0] == inst:
                if i[3] == np.inf:
                    laser_inst[i[2]]=0
                else:
                    laser_inst[i[2]]=i[3]                
                #laser_angle.append(i[2])
                #laser_range.append(i[3])
            else:
                if self.add_time:
                    laser_inst["time"] = i[1]
                
                #laser_pd_series= pd.Series([laser_angle,laser_range], name=inst)
                #laser_list= pd.concat([laser_list,laser_pd_series], axis=1) 

                laser_inst.name=inst

                laser_list=pd.concat([laser_list, laser_inst.copy()],axis=1)

                target_inst["pos_x"] = i[4]
                target_inst["pos_y"] = i[5]
                target_inst["pos_yaw"] = i[6]
                target_inst.name= inst
                target_list=pd.concat([target_list,target_inst],axis=1)

                #laser_angle=[]
                #laser_range=[]

                #laser_angle.append(i[2])
                #laser_range.append(i[3]) 
                
                if i[3] == np.inf:
                    laser_inst[i[2]]=0
                else:
                    laser_inst[i[2]]=i[3]     

                inst+=1

        self.features = laser_list.T
        self.targets = target_list.T

    def divide_data(self, test_r):
        self.TR_features, self.TS_features, self.TR_targets, self.TS_targets = train_test_split(self.features, self.targets, test_size= test_r, random_state= 42)

