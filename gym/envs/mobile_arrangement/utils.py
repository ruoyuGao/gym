import cv2
import spfa
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

BEV_PIXEL_WIDTH = 160
ActionMap_PIXEL_WIDTH = 160 
ActionMap_WIDTH = 0.12
ActionMap_PIXELS_PER_METER = ActionMap_PIXEL_WIDTH / 0.12
BEV_PIXELS_PER_METER = BEV_PIXEL_WIDTH / 0.12
distance_threshold = 0.0005
min_move_distance_threshole = 2.5 ### in pixel cordinate
COLOR_threshold=90
Shortest_path_channel_scale = 2

def closest_valid_cspace_indices(i, j, configuration_space):
    closest_cspace_indices = distance_transform_edt(1 - configuration_space, return_distances=False, return_indices=True)
    return closest_cspace_indices[:, i, j]

def create_global_shortest_path_map(robot_position , configuration_space, LOCAL_STEP_LIMIT):
    pixel_i, pixel_j = robot_position[0], robot_position[1]
    pixel_i, pixel_j = closest_valid_cspace_indices(pixel_i, pixel_j,configuration_space)
    global_map, _ = spfa.spfa(configuration_space, (pixel_i, pixel_j))
    global_map /= BEV_PIXELS_PER_METER
    global_map /= (np.sqrt(2) * BEV_PIXEL_WIDTH) / BEV_PIXELS_PER_METER
    
    global_map[global_map==0] = 1
    global_map[robot_position[0],robot_position[1]] = 0

    global_map *= Shortest_path_channel_scale
    global_map -= 1 

    global_map=np.expand_dims(global_map, axis=2)
    return global_map

def position_to_pixel_indices(position_x, position_y, image_shape):
    pixel_i = np.floor(image_shape[0] / 2 - position_y * BEV_PIXELS_PER_METER).astype(np.int32)
    pixel_j = np.floor(image_shape[1] / 2 + position_x * BEV_PIXELS_PER_METER).astype(np.int32)
    pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
    pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
    return pixel_i, pixel_j

def pixel_indices_to_position(pixel_i, pixel_j, image_shape):
    position_x = (pixel_j - image_shape[1] / 2) / BEV_PIXELS_PER_METER
    position_y = (image_shape[0] / 2 - pixel_i) / BEV_PIXELS_PER_METER
    return 100*position_x+6, 6-100*position_y

def create_padded_room_zeros():
    return np.zeros((
            int(2 * np.ceil((0.12 * BEV_PIXELS_PER_METER) / 2)),  # Ensure even
            int(2 * np.ceil((0.12 * BEV_PIXELS_PER_METER) / 2))
        ), dtype=np.float32)

def create_padded_room_BEV():
    return 0.420*np.ones((
            int(2 * np.ceil((0.12 * BEV_PIXELS_PER_METER) / 2)),  # Ensure even
            int(2 * np.ceil((0.12 * BEV_PIXELS_PER_METER) / 2)),3
        ), dtype=np.float32)

def convert2image():
    img=create_padded_room_zeros()
    img[0,:]=1
    img[-1,:]=1
    img[:,0]=1
    img[:,-1]=1    
    return img

def gen_transformationM(previous_transformation,action,value):
    action_M=np.identity(4)
    if action==0:
        rotate_angle=-np.pi/180*value
        action_M[0,0]=np.cos(rotate_angle)
        action_M[0,2]=np.sin(rotate_angle)
        action_M[2,0]=-np.sin(rotate_angle)
        action_M[2,2]=np.cos(rotate_angle)
        return np.matmul(previous_transformation,action_M)
    elif action==1:
        rotate_angle=np.pi/180*value
        action_M[0,0]=np.cos(rotate_angle)
        action_M[0,2]=np.sin(rotate_angle)
        action_M[2,0]=-np.sin(rotate_angle)
        action_M[2,2]=np.cos(rotate_angle)
        return np.matmul(previous_transformation,action_M)
    elif action==2:
        move_dis = value*1e-2
        action_M[2,3]=move_dis
        return np.matmul(previous_transformation,action_M)
    elif action==3:
        move_dis = -value*1e-2
        action_M[2,3]=move_dis
        return np.matmul(previous_transformation,action_M)

def pad_image(img):
    img_=img.copy()
    img_shape=img.shape[0]
    pad=15
    mask=np.where(img[pad:img_shape-pad,pad:img_shape-pad]==1)
    directions = [[1, 0],[-1, 0], [0,1], [0,-1],[1,1],[-1,1],[1,-1],[-1,-1],\
     [2, 0], [-2, 0], [0,2], [0,-2],[2,2],[-2,2],[2,-2],[-2,-2]]  # dir you want to go
    for dir in directions:
        img_[mask[0]+dir[0]+pad,mask[1]+dir[1]+pad]=1
    return img_

def distance(position1, position2):
    return ((position1[0]-position2[0])**2+(position1[1]-position2[1])**2)**0.5