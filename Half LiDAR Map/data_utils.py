from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def uniform_rotation(db):
    unidir_map = []
    counter = 0
    buffer = []
    corr_angle = 0

    for i in db:
        i[2] = np.around((((i[2]+i[6]) + np.pi) % (2*np.pi))-np.pi,decimals=5)
        i[6] = 0
        buffer.append(i)
        if counter == 719:
            if corr_angle != 0:
                for j in range(720):
                    buffer[(j+corr_angle)%720][2]=np.radians((j-360)/2)

            buffer.sort(key= lambda i:i[2])
            unidir_map.append(buffer.copy())
            counter = 0
            buffer = []
        else:
            counter += 1

    return unidir_map

def pnt_on_grid(map_img, pnt, rad=1, map_scale=0.05):
    #rad = 1 esatta
    map_grid = np.zeros(map_img.shape)

    coord_x = pnt[0] / map_scale
    coord_y = pnt[1] / map_scale

    px_mod = 0

    for r in range(rad):
        for i in range(360):
            x=int(np.cos(np.radians(i))*r)
            y=int(np.sin(np.radians(i))*r)

            int_x = int(coord_x+x)
            int_y = int(coord_y+y)

            if int_x >= 0 and int_x < map_grid.shape[1] and int_y >= 0 and int_y < map_grid.shape[0] and map_grid[int_y][int_x]==0:
                map_grid[int_y][int_x] = 1
                #map_grid[int_y][int_x] = 1/(1+np.log(1+(np.abs(x)+np.abs(y))))
                #px_mod += 1/(1+np.log(1+(np.abs(x)+np.abs(y))))

    #normalize
    #for i in range(map_grid.shape[0]):
    #    for j in range(map_grid.shape[1]):
    #        if map_grid[i][j]!= 0:
    #            map_grid[i][j] = map_grid[i][j]/px_mod

    return map_grid

def quant_rot(rot_value, num_intervals, rad=1):
    result = np.zeros(num_intervals)

    rot_value += np.pi

    array_pos = int((rot_value/(2*np.pi))*num_intervals)
    
    for i in range(rad):
        result[((array_pos + i) % num_intervals)] = 1
        result[((array_pos - i) % num_intervals)] = 1

    return result

#function of the creation of the instace LiDAR

def create_lidar(x, y, r, map, map_scale=0.05):
    #correzione assi
    #x=map.shape[0]-x-1
    #r=r-180

    #if the position of the image is not a free space (obstacle or unknown) it returns None
    if map[x][y] != 254:
        return None
    else:
        #create an artificial LiDAR based on the neibourhood pixels
        result = look_around(map,x,y,r,int(20/map_scale))
        return result

def look_around(map,x,y,r,dist,map_scale=0.05):
    result = {}

    for i in range(360):
        angle = ((i + r) % 360)
        lidar_range = get_range(map, x, y, np.radians(angle),dist)
        result["range{}".format(i)] = lidar_range*map_scale
        result["angle{}".format(i)] = np.around(np.radians((i-180)),decimals=5)  

    return result

def get_range(map,x,y,rad,dist):

    for i in range(dist):
        x_range = np.cos(rad) * i
        y_range = np.sin(rad) * i
        if map[x+int(x_range)][y+int(y_range)] != 254:
            break

    if map[x+int(x_range)][y+int(y_range)] == 0:
        
        return min(np.sqrt((np.power(int(x_range),2))+(np.power(int(y_range),2))), dist)
    
    return dist