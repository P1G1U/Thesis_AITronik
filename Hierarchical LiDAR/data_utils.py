from random import randint
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import gridspec
from matplotlib.patches import Wedge
from skimage.transform import rescale

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

def pnt_on_grid(map_img, pnt, rad=1, map_scale=0.05, x_0=0, y_0=0):
    #rad = 1 esatta
    map_grid = np.zeros(map_img.shape)

    coord_x = (pnt[0] - x_0) / map_scale
    coord_y = (pnt[1] - y_0) / map_scale

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

def background_map(mask, map_bg):

    if mask.shape[0] == map_bg.shape[0] and mask.shape[1] == map_bg.shape[1]:

        for row in range(mask.shape[0]):
            for column in range(mask.shape[1]):
                if mask[row][column] == 1:
                    map_bg[row][column] = 50

        return map_bg
    
    return None

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


def display(display_list, rot_list, name="test"):

    subtitle=["True Mask", "Predicted Mask"]

    gs = gridspec.GridSpec(20, 2) 

    fig = plt.figure(figsize=(13,7))

    mask_1 = plt.subplot(gs[:13,0])
    mask_2 = plt.subplot(gs[:13,1])
    rot_1 = plt.subplot(gs[13:,0])
    rot_2 = plt.subplot(gs[13:,1])

    mask_1.set_title(subtitle[0])
    mask_1.imshow(tf.keras.utils.array_to_img(display_list[0]))
    mask_1.axis("off")
    
    mask_2.set_title(subtitle[1])
    mask_2.imshow(tf.keras.utils.array_to_img(display_list[1]))
    mask_2.axis("off")

    draw_circle(fig, rot_1, rot_list[0])
    draw_circle(fig, rot_2, rot_list[1])

    plt.subplots_adjust(hspace=0)
    plt.savefig("img/{}.png".format(name))
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def draw_circle(fig, ax, array_rot):
   radius = 1.5
   center = (0, 0)

   w = []

   for r in range(90):
      theta1, theta2 = r*4, r*4 + 4
      if array_rot[r] == 1:
         w_ = Wedge(center, radius, theta1, theta2, fc='black', edgecolor='black')
      else:
         w_ = Wedge(center, radius, theta1, theta2, fc='white', edgecolor='grey')

      w.append(w_)

   for wedge in w:
      ax.add_artist(wedge)

   ax.axis('equal')
   ax.set_xlim(-5, 5)
   ax.set_ylim(-5, 5)
   ax.axis("off")

def get_fingerprint(map_img, path, rad):
    
    map_grid = np.zeros(map_img.shape)

    for pnt in path:

        coord_x = pnt[0] / 0.05
        coord_y = pnt[1] / 0.05

        for r in range(rad):
            for i in range(360):
                x=int(np.cos(np.radians(i))*r)
                y=int(np.sin(np.radians(i))*r)

                int_x = int(coord_x+x)
                int_y = int(coord_y+y)

                if int_x >= 0 and int_x < map_grid.shape[1] and int_y >= 0 and int_y < map_grid.shape[0] and map_grid[int_y][int_x]==0:
                    map_grid[int_y][int_x] = 1

    return map_grid

def get_pos_from_list_fp(list_fp):
    
    pos_x = np.array(list_fp)[:,1]
    pos_y = np.array(list_fp)[:,0]

    pos_x = np.average(pos_x)
    pos_y = np.average(pos_y)

    return [pos_x,pos_y]

def get_conf_from_list_fp(list_fp):

    center = get_pos_from_list_fp(list_fp)

    error_from_center = []

    for pnt in list_fp:
        error = np.sqrt(np.power((center[0]-pnt[0]),2)+np.power((center[1]-pnt[1]),2))
        error_from_center.append(error)

    return np.std(error_from_center)

def get_rot_from_slice(list_fp,tot_slice):

    avarage = 0

    for i in list_fp:
        avarage = avarage + i

    avarage = (avarage / len(list_fp))

    avarage = corr_angle_deg(avarage, 360/tot_slice)

    return avarage

def get_conf_from_slice(list_fp,tot_slice):

    center = get_rot_from_slice(list_fp,tot_slice)

    error_from_center = []

    for pnt in list_fp:

        error = ((pnt * (360/tot_slice))+(180/tot_slice) -center) -180

        error_from_center.append(error)

    return np.std(error_from_center)

def list_from_fingerprint(map_fingerprint, pixel_scale=0.05, mask=False):
    
    trail_list = []

    heigth = map_fingerprint.shape[0]
    width = map_fingerprint.shape[1]

    if mask:
        for i in range(heigth):
            if map_fingerprint[i].any():
                for j in range(width):
                    if map_fingerprint[i][j] == 1:
                        trail_list.append([i*pixel_scale,j*pixel_scale])
    
    else: 
        for i in range(heigth):
            if tf.argmax(map_fingerprint[i],axis=-1).numpy().any():
                for j in range(width):
                    if tf.argmax(map_fingerprint[i][j]).numpy() == 1:
                        trail_list.append([i*pixel_scale,j*pixel_scale])

    return trail_list

def list_from_rotation(cake):
    list_slice = []

    #up = False
    #down = False

    for i in range(len(cake)):
        if tf.argmax(cake[i]).numpy() == 1:
            list_slice.append(i-(len(cake)/2))
    
        #if i > len(cake)-3:
        #    down = True
        #if i<2:
        #    up = True

    #if up and down:
        #for i in range(len(list_slice)):
        #    if list_slice[i] < 10:
        #        list_slice[i] += len(cake)

    break_array = np.zeros(len(list_slice))

    if len(list_slice) != 0:

        while max(list_slice)-min(list_slice) > len(cake)/2:
            if break_array[np.argmin(list_slice)] == 1:
                break
            break_array[np.argmin(list_slice)] = 1
            list_slice[np.argmin(list_slice)] += len(cake)
        

    return list_slice

def get_fingerprint(map_img, path, rad, pixel_scale=0.05):
    
    map_grid = np.zeros(map_img.shape)

    for pnt in path:

        coord_x = pnt[0] / pixel_scale
        coord_y = pnt[1] / pixel_scale

        for r in range(rad):
            for i in range(360):
                x=int(np.cos(np.radians(i))*r)
                y=int(np.sin(np.radians(i))*r)

                int_x = int(coord_x+x)
                int_y = int(coord_y+y)

                if int_x >= 0 and int_x < map_grid.shape[1] and int_y >= 0 and int_y < map_grid.shape[0] and map_grid[int_y][int_x]==0:
                    map_grid[int_y][int_x] = 1

    return map_grid

def corr_angle_rad(rad_angle):
    
    while rad_angle < (-np.pi/2) or rad_angle > (np.pi/2):
        if rad_angle < (-np.pi/2):
            rad_angle += np.pi
        else:
            rad_angle -= np.pi

    return rad_angle

def corr_angle_deg(deg_angle,disc=1):
    
    while deg_angle < (-180/disc) or deg_angle > (180/disc):
        if deg_angle < (-180/disc):
            deg_angle += 360/disc
        else:
            deg_angle -= 360/disc

    return deg_angle

def zoom_in(map_img, pos, zoomed_width, zoomed_height, scale=1):

    pixel_scale = 0.1 / scale

    map_img = rescale(map_img,scale)

    height = map_img.shape[0]
    width = map_img.shape[1]

    tot_pixel = height * width

    max_pixel =  zoomed_height * zoomed_width

    if max_pixel > tot_pixel:
        return map_img

    if (pos[0]/pixel_scale) >= height or (pos[1]/pixel_scale) >= width:
        return None

    x_0 = int((pos[1]/pixel_scale) - (zoomed_width/2) + randint(-int(2*(zoomed_width/5)),int(2*(zoomed_width/5))))
    corr_x = 0
    
    if x_0 + zoomed_width > width:
        corr_x = width - (x_0 + zoomed_width) - 1

    if x_0 < 0:
        corr_x = - (x_0)
    
    x_0 = x_0 + corr_x

    y_0 = int((pos[0]/pixel_scale) - (zoomed_height/2) + randint(-int(2*(zoomed_height/5)),int(2*(zoomed_height/5))))
    corr_y = 0

    if y_0 + zoomed_height > height:
        corr_y = height - (y_0 + zoomed_height) - 1

    if y_0 < 0:
        corr_y = - (y_0)
    
    y_0 = y_0 + corr_y
    
    map_zoomed = np.zeros((zoomed_height,zoomed_width))

    for i in range(zoomed_height):
        for j in range(zoomed_width):
            map_zoomed[i][j] = map_img[i+y_0][j+x_0]
    
    return map_zoomed, x_0, y_0