"""Define the class of Data Access Object that formats the data read from the files"""

import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np

class DAO:

    laser_db = None

    features = None
    targets = None

    TR_features = None
    TS_features = None
    TR_targets = None
    TS_targets = None

    TR_features_unidir = None
    TS_features_unidir = None
    TR_targets_unidir = None
    TS_targets_unidir = None

    TR_targets_pos = None
    TS_targets_pos = None
    TR_targets_rot = None
    TS_targets_rot = None

    TR_targets_unidir_pos = None
    TS_targets_unidir_pos = None
    TR_targets_unidir_rot = None
    TS_targets_unidir_rot = None

    add_time = False

    """Initialization of the object: filename is the file .csv with the data stored, rate is the ratio test/total data, time is a bool variable for the usage of timestamp of the istance"""
    def __init__(self, filename="", rate = 0.25, time=False):
        self.filename = filename
        self.add_time = time
        self.test_r = rate

    """read: create the array of the features and of the targets"""
    def read(self):
        
        with open("data/"+self.filename, "r") as data:
            self.laser_db = pd.read_csv(data, delimiter=";")

        self.features, self.targets = self.elaborate(self.laser_db)

    """elaborate: parse the file and save the istance of the laser with their real position and rotation"""
    def elaborate(self, db):

        fill_value = 20 #the value that is used for the inf range of the LiDAR istance

        laser_list= pd.DataFrame()
        target_list= pd.DataFrame()
        inst = 1 #counter starts from 1
        new = True

        laser_inst=pd.Series(dtype="float32")
        target_inst=pd.Series(dtype="float32")

        if len(db.values) > (5250*360):
            db = db.truncate(before=0, after=(5250*360))

        for i in db.values:

            if not new:
                if i[0] == inst:
                    if i[3] == np.inf:
                        laser_inst["range{}".format(laser_id)]=fill_value
                        #laser_inst["angle{}".format(laser_id)]=i[2]
                        #laser_inst[i[2]]=fill_value
                    else:
                        laser_inst["range{}".format(laser_id)]=i[3]
                        #laser_inst["angle{}".format(laser_id)]=i[2]
                        #laser_inst[i[2]]=i[3]                
                else:
                    if self.add_time:
                        laser_inst["time"] = i[1]
                    
                    laser_inst.name=inst

                    laser_list=pd.concat([laser_list, laser_inst.copy()],axis=1) 

                    inst+=1

                    new = True
                    
            if new:
                target_inst["pos_x"] = i[4]
                target_inst["pos_y"] = i[5]
                target_inst["pos_yaw"] = i[6]
                target_inst.name= inst
                target_list=pd.concat([target_list,target_inst],axis=1)

                laser_id = 0

                if i[3] == np.inf:
                    laser_inst["range{}".format(laser_id)]=fill_value
                    #laser_inst["angle{}".format(laser_id)]=i[2]
                    #laser_inst[i[2]]=fill_value
                else:
                    laser_inst["range{}".format(laser_id)]=i[3]
                    #laser_inst["angle{}".format(laser_id)]=i[2]
                    #laser_inst[i[2]]=i[3]   
            
                new = False
                
            laser_id+=1

        if self.add_time:
            laser_inst["time"] = i[1]

        laser_inst.name=inst

        laser_list=pd.concat([laser_list, laser_inst.copy()],axis=1) 

        return (laser_list.T, target_list.T)

    """divide_data: takes the features and the targets of the objects and split in train and test data with the rate test/total data"""
    def divide_data(self):
        self.TR_features, self.TS_features, self.TR_targets, self.TS_targets = train_test_split(self.features, self.targets, test_size= self.test_r, random_state= 42)

    """uniform_rotation: tranform the LiDAR istance of the db to 0 yaw and return the features and the targets transformed, splitted in train and test"""
    def uniform_rotation(self, db):
        
        unidir_map_list = []
        counter = 0
        buffer = []
        corr_angle = 0

        for i in db.values:
            i[2] = np.around((((i[2]+i[6]) + np.pi/2) % (np.pi))-np.pi/2,decimals=5)
            i[6] = 0
            buffer.append(i)
            if counter == 359:
                if corr_angle != 0:
                    for j in range(360):
                        buffer[(j+corr_angle)%360][2]=np.radians((j-180))

                buffer.sort(key= lambda i:i[2])

                for j in buffer:
                    unidir_map_list.append(j)

                counter = 0
                buffer = []
            
            else:
                counter += 1

        unidir_map = pd.DataFrame(unidir_map_list,columns=["cnt","time","angle","range","pos_x","pos_y","pos_yaw"])

        features, targets = self.elaborate(unidir_map)

        self.TR_features_unidir, self.TS_features_unidir, self.TR_targets_unidir, self.TS_targets_unidir = train_test_split(features, targets, test_size= self.test_r, random_state= 42)
