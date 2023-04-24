import os
import time
import random
import math
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2
from scipy.ndimage import center_of_mass, rotate
from typing import Optional
from gym.envs.mobile_arrangement.utils import *
import copy
import random

dir_path = os.path.dirname(os.path.realpath(__file__))

BEV_PIXEL_WIDTH = 160
ActionMap_PIXEL_WIDTH = 160 
ActionMap_WIDTH = 0.12
ActionMap_PIXELS_PER_METER = ActionMap_PIXEL_WIDTH / 0.12
BEV_PIXELS_PER_METER = BEV_PIXEL_WIDTH / 0.12
distance_threshold = 0.0005
min_move_distance_threshole = 2.5 ### in pixel cordinate
COLOR_threshold=90
Shortest_path_channel_scale = 2

class BoxWorld(gym.Env):
    def __init__(self, isGUI: Optional[bool] = False):
        '''
        This class is our gym environment of a Box world with random objects placed inside
        self.action_space: Our robot's action space
        self.observation_space: Our robot's observation space (unsure whether pose or visuals)
        self.client: Our PyBullet physics client
        self.plane: The Box World env's plane floor
        self.spawn_height: The default height of each agent/object when spawned
        self.robots: Our four robots
        self.poses: Our four robots' initialization positions
        self.grab_distance: The distance that measures each robot's reach
        self.length: The length of each side of the box and also the length of the wall
        self.width: The width of only the wall
        self.objects: A list to keep track of the random objects we place
        self.num_objects: The number of random objects placed into the box
        self.num_objects: The initialized object positions
        self.img_w: Width of the camera
        self.fov: Scope of the camera
        self.distance: Idk what this is
        self.fpv_curr
        self.load_steps: The number of steps simulation takes to stepSimulation
        self.backpack
        self.force_scalar: The scalar multipled to the forces applied to the robots movement
        ## The remaining are object files
        '''

        self.client = p.connect(p.GUI) if isGUI else p.connect(p.DIRECT)
        p.setTimeStep(1./240, self.client)
        self.plane = None
        self.spawn_height = 0.5
        self.num_robots = 1
        self.robot = None
        self.init_pos = [6,6,self.spawn_height]
        self.length = 12
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(80, 80, 6), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(6)
        self.reward_space = gym.spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32)
        self.width = 0.1
        self.wall_height = 4
        self.objects = []
        self.color_index = []
        self.num_objects = 3
        self.object_init_pos = []
        self.backpack = [(-1, ""), -1]
        self.num_correct_drops_for_obj = [0]*self.num_objects
        self.num_correct_drops = 0
        self.prev_picked_color = None
        #self.img_w = 160
        self.img_w = 80
        self.obs_robot_init = np.zeros((self.img_w, self.img_w, self.img_w, 4), dtype=np.float32)
        self.fov = 90
        self.distance = 20
        self.fpv_prev = self.obs_robot_init
        self.fpv_curr = self.obs_robot_init
        self.fpv_depth = np.zeros((self.img_w, self.img_w), dtype=np.float32)
        self.bev = None
        self.shortpath_BEV = None
        self.shortpath_from_agent = None
        self.short_path_from_target_object = []
        self.kernel = np.ones((15,15),np.uint8)
        self.check_move_or_not = False
        self.load_steps = 100
        self.max_episode_steps = 1000
        self.successful_picks = 0
        self.LOCAL_STEP_LIMIT = 30
        self.agent_name = os.path.join(os.path.dirname(__file__), '../resources/agent.urdf')
        self.sphere_name = os.path.join(os.path.dirname(__file__), '../resources/sphere2.urdf')
        self.cube_name = os.path.join(os.path.dirname(__file__), '../resources/cube.urdf')
        self.cylinder_name = os.path.join(os.path.dirname(__file__), '../resources/cylinder.urdf')
        self.cone_name = os.path.join(os.path.dirname(__file__), '../resources/obj_files/cone_blue.obj')
        self.plane_name = os.path.join(os.path.dirname(__file__), '../resources/plane.urdf')

    def reset(self):
        """
        Reset the environment, place robot in (6,6) and randomly initiale objects within box
        """
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        #p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) # take this away for official use but for debug it's nice bc it slows animation
        p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-90, cameraTargetPosition=[self.length/2,self.length/2,0])
        #p.resetDebugVisualizerCamera(cameraDistance=20, cameraYaw=0, cameraPitch=-90, cameraTargetPosition=[self.length/2,self.length/2,0])
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1) # see hitboxes + makes things faster

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        self.backpack = [(-1, ""), -1]
        #self.pre_pos = np.array([6,6,0])
        self.pre_pos = [6,6,0]
        #self.curr_pos = np.array([6,6,0])
        self.curr_pos = [6,6,0]
        self.objects = []
        self.color_index = []
        self.object_init_pos = []
        self.successful_picks = 0
        self.fpv_prev = self.obs_robot_init
        self.fpv_curr = self.obs_robot_init
        self.fpv_depth = np.zeros((self.img_w, self.img_w), dtype=np.float32)
        self.step_count = 0
        self.num_correct_drops = 0
        self.prev_picked_color = None
        self.fpv_depth = np.zeros((self.img_w, self.img_w), dtype=np.float32)
        self.bev = None
        self.shortpath_BEV = None
        self.shortpath_from_agent = None
        self.short_path_from_target_object = []
        self.kernel = np.ones((15,15),np.uint8)
        self.one_hot = [0] * self.num_objects

        self.plane = p.loadURDF(fileName=self.plane_name, basePosition=[self.length/2,self.length/2,0])
        self.spawn_wall()
        self.robot = p.loadURDF(fileName=self.agent_name, basePosition=self.init_pos)
        self.spawn_objects(init=True)
        
        self.load_camera()
        self.bev_target = self.bev
        self.spawn_objects() #rerandomize locations
       
        #self.step([-1, 0])
        self.load_camera()
        self.fpv_prev = self.fpv_curr # on first step, these should be same

        self.shortpath_BEV = convert2image()
        img_path = cv2.dilate(self.shortpath_BEV, self.kernel, iterations=1)
        self._shortpath_from_agent= 1 - img_path
        self.short_path_from_target_object = []

        color_onehot = np.eye(6)[self.color_ind]
        type_onehot = np.eye(4)[self.objs_types_index]
        self.color_and_type_binary_vect = np.concatenate((color_onehot,type_onehot),axis=1)

        state = self.get_state(init=True)
        self.Current_step=0 
        
        # self.visualize(target=True) # for human gui, comment out for real
        # self.visualize() # for human gui, comment out for real
        self.pre_dis_to_target = self.compute_configuration_distance()

        # color_onehot = np.eye(6)[self.color_ind]
        # type_onehot = np.eye(4)[self.objs_types_index]
        # self.color_and_type_binary_vect = np.concatenate((color_onehot,type_onehot),axis=1)

        # color_and_type_binary_vect = np.concatenate((color_onehot,type_onehot),axis=1)
        # return state, color_and_type_binary_vect

        return state

    def create_object_spm(self):
        """
        Creates target object spms
        """
        for i in range(self.num_objects):
            pos = self.object_init_pos[i]
            entity_pos_y, entity_pos_x = position_to_pixel_indices((pos[0]-6)/100,(6-pos[1])/100,[BEV_PIXEL_WIDTH,BEV_PIXEL_WIDTH])
            short_path_from_target_object = create_global_shortest_path_map(robot_position = [entity_pos_y,entity_pos_x],\
                configuration_space=self._shortpath_from_agent,LOCAL_STEP_LIMIT=self.LOCAL_STEP_LIMIT)
            self.short_path_from_target_object.append(self.trans(short_path_from_target_object))

        # (3, 160, 160)
        self.output_spm = np.concatenate((self.short_path_from_target_object[0],\
                            self.short_path_from_target_object[1],\
                            self.short_path_from_target_object[2]), axis=0).flatten()

    def get_state(self, init=False):
        """
        Get current state
        input: init=T/F for creating state for target or any other step
        output: returns list of agent pos, one_hot for pick/drop, obj pcds, target spm, agent spm
        """
        # spm
        if init: # init pos
            self.create_object_spm()
        # state=[self.pre_pos, self.curr_pos,\
        #     self.fpv_prev, self.fpv_curr, self.fpv_depth, self.output_spm, self.get_object_gtpose(), self.one_hot]
        # state=[self.pre_pos, self.curr_pos,\
        #     self.fpv_prev, self.fpv_curr, self.fpv_depth, self.output_spm, self.get_object_gtpose(), \
        #     self.one_hot, self.color_and_type_binary_vect, self.backpack_onehot()]
        # only return the first person image and top down image
        assert self.bev_target is not None, "target image(bev_target) should not be none"
        state = [self.fpv_prev, self.bev_target]
        # print(self.pre_pos.shape)   # (3,)
        # print(self.curr_pos.shape)  # (3,)
        # print(self.fpv_prev.shape)  # (80,80,3)
        # print(self.fpv_curr.shape)  # (80,80,3)
        # print(self.fpv_depth.shape) # (80,80)
        # print(self.output_spm.shape)  # (3,160,160)
        return state

    def generate_masked_pcd(self, fpv_curr, fpv_depth):
        """
        Created the masked point cloud data
        input: fpv_curr = current fpv image; fpv_depth = current fpvd image
        output: three 1x4 PCD arrays
        """
        fpv_rgba = fpv_curr.reshape(self.img_w,self.img_w,4)
        fpv = cv2.cvtColor(fpv_rgba, cv2.COLOR_RGBA2RGB)
        depth = fpv_depth.reshape(self.img_w,self.img_w)
        color_filters = [[np.array([160, 0, 0]), np.array([255, 0, 0])], [np.array([160, 160, 0]), np.array([255, 255, 0])], \
            [np.array([0, 160, 0]), np.array([0, 255, 0])],[np.array([0, 0, 160]), np.array([0, 0, 255])]]
        # red yellow green blue
        fx = 40 
        cx = 40
        fy = 40
        cy = 40
        output = []
        for i in range(self.num_objects):
            idx = self.color_index[i]
            mask = cv2.inRange(fpv, color_filters[idx][0], color_filters[idx][1])
            if sum(sum(mask)) == 0: 
                cx, cy, cz = 0, 0, 20.0 
                pcdx = (cx-cx)/fx*cz/12
                pcdy = (cy-cy)/fy*cz/12
                pcdz = cz/12
            else:
                cy, cx = center_of_mass(mask)
                cy, cx = int(cy), int(cx)
                cz = depth[cy,cx]
                pcdx = (cx-cx)/fx*cz/12
                pcdy = (cy-cy)/fy*cz/12
                pcdz = cz/12
            tmp = np.array([[pcdx, pcdy, pcdz,1]])
            output.append(tmp)
        return output

    def visualize(self, target=False):
        """
        Save BEV, FPV(D), SPM, PCD images
        input: state = the current state; target = T/F whether target visualization or visualization after step
        """

        # self.fpv_prev = self.fpv_curr # (25600,)
        # self.fpv_curr = fpv # (25600,)
        # self.fpv_depth = fpv_depth  # (6400,)
        # self.bev = tdv # (102400,)

        if target:
            target_bev = np.reshape(np.clip(self.bev_target, 0, 255), (160,160,4)).astype(np.uint8)
            f, axarr = plt.subplots(1,4)
            axarr[0].imshow(target_bev)
            axarr[0].axis('off')
            axarr[1].imshow(np.flip(np.moveaxis(self.short_path_from_target_object[0], 0, -1), 1))
            axarr[1].axis('off')
            axarr[2].imshow(np.flip(np.moveaxis(self.short_path_from_target_object[1], 0, -1), 1))
            axarr[2].axis('off')
            axarr[3].imshow(np.flip(np.moveaxis(self.short_path_from_target_object[2], 0, -1), 1))
            axarr[3].axis('off')
            filename = dir_path + "/imgs/target.png"
            plt.savefig(filename)
            plt.close()
            return
        bev = np.reshape(np.clip(self.bev, 0, 255), (160,160,4)).astype(np.uint8)
        fpv_prev = np.reshape(np.clip(self.fpv_prev, 0, 255), (self.img_w,self.img_w,3)).astype(np.uint8)
        fpv_curr = np.reshape(np.clip(self.fpv_curr, 0, 255), (self.img_w,self.img_w,3)).astype(np.uint8)
        f, axarr = plt.subplots(1,4) 
        axarr[0].imshow(bev)
        axarr[0].axis('off')
        axarr[1].imshow(fpv_prev)
        axarr[1].axis('off')
        axarr[2].imshow(fpv_curr)
        axarr[2].axis('off')
        axarr[3].imshow(np.reshape(self.fpv_depth, (self.img_w,self.img_w)))
        axarr[3].axis('off')
        filename = dir_path + "/imgs/timestep" + str(self.Current_step) +".png"
        plt.savefig(filename)
        plt.close()
    
    def compute_configuration_distance(self):
        """
        find distance between curr obj pos and targ pos
        output: return sum of distances
        """
        dis = 0 
        for i in range(self.num_objects):
            if i == self.backpack[1]: continue
            target_entity_pos = self.object_init_pos[i]
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            entity_pos = [pos[0], pos[1], pos[2]]
            dis_ = distance([target_entity_pos[0],target_entity_pos[1]],[entity_pos[0],entity_pos[1]])
            dis += dis_
        return dis

    def trans(self,image):
        """
        transform for pytorch tensor
        input: image * image *3
        out: 3*image*image
        """
        out = np.moveaxis(image, -1, 0)  
        return out

    def load_camera(self):
        """
        Take one frame of robot fpv and bev
        """
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.robot)

        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        xA, yA, zA = agent_pos
        #zA = zA + 0.3 # change vertical positioning of the camera

        xB = xA + math.cos(yaw) * self.distance
        yB = yA + math.sin(yaw) * self.distance
        zB = zA

        view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[xA, yA, zA],
                            cameraTargetPosition=[xB, yB, zB],
                            cameraUpVector=[0, 0, 1.0])

        projection_matrix = p.computeProjectionMatrixFOV(
                                fov=self.fov, aspect=1.5, nearVal=0.02, farVal=self.distance)

        ## WE CAN ENABLE/DISABLE SHADOWS HERE
        robot_fpv = p.getCameraImage(self.img_w, self.img_w,
                                view_matrix,
                                projection_matrix, 
                                flags=p.ER_NO_SEGMENTATION_MASK)[2:4]

        bev_view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[6, 6, 10],
                            cameraTargetPosition=[6, 5.99, 0.5],
                            cameraUpVector=[0,0,1])

        bev_projection_matrix = p.computeProjectionMatrixFOV(
                        fov=90, aspect=1, nearVal=0.02, farVal=100)

        bev = p.getCameraImage(BEV_PIXEL_WIDTH, BEV_PIXEL_WIDTH,
                            bev_view_matrix, bev_projection_matrix,
                            flags=p.ER_NO_SEGMENTATION_MASK)[2]

        seg_mask = set(list(p.getCameraImage(self.img_w, self.img_w,
                                view_matrix,
                                projection_matrix)[-1].flatten()))
        self.one_hot = [0] * self.num_objects
        for i in range(len(self.objects)):
            if self.objects[i][0] in seg_mask:
                self.one_hot[i] = 1

        fpv = cv2.cvtColor(np.array(robot_fpv[0], dtype=np.float32), cv2.COLOR_RGBA2RGB).flatten() # 80x80x4 (RGBA) might want to change this to RGB
        fpv_depth = np.array(robot_fpv[1], dtype=np.float32).flatten() # 80x80
        tdv = np.array(bev, dtype=np.float32).flatten() # 160x160x4


        self.fpv_prev = self.fpv_curr # (19200,)
        self.fpv_curr = fpv # (19200,)
        self.fpv_depth = fpv_depth  # (6400,)
        self.bev = tdv # (76800,)

    def step(self, action, turn_step=10):
        """
        Env takes one step, one timestep for robot movement and one frame for cameras
        input: action = integer able to fulfill one of these conditions
        output: next state
        """

        # 0 is pick, 1 is drop
        #action_, pick_or_drop = action
        self.fwd_step, fwd_drift = np.random.normal(0.15, 0.01), np.random.normal(0, 0)
        #self.fwd_step, fwd_drift = np.random.normal(2, 0.01), np.random.normal(0, 0) # for testing to see steps
        self.turn_step = np.random.normal(turn_step, 1)

        self.Current_step += 1
        rewards = 0
        done = False
        info = None

        #self.action_map = np.unravel_index(action_, (BEV_PIXEL_WIDTH, BEV_PIXEL_WIDTH))
        #target_y, target_x = pixel_indices_to_position(self.action_map[1],self.action_map[0],[BEV_PIXEL_WIDTH,BEV_PIXEL_WIDTH])
        #self.agent_pos_y, self.agent_pos_x = position_to_pixel_indices((self.curr_pos[0]-6)/100,(6-self.curr_pos[1])/100,[BEV_PIXEL_WIDTH,BEV_PIXEL_WIDTH])
        #self.pre_pos = self.curr_pos
        #self.curr_pos[0] = target_x
        #self.curr_pos[1] = target_y

        # say 0,1,2,3,4,5; move forward, back, turn left, turn right, pickup, drop
        #print(action)
        if action == 0: # move forward
            #self.check_move_or_not = self.move_agent(self.fwd_step, fwd_drift)
            self.check_move_or_not = self.move_agent(self.fwd_step, fwd_drift)
        if action == 1: # move backward
            self.check_move_or_not = self.move_agent(-self.fwd_step, fwd_drift)
        if action == 2: # turn left
            self.turn_agent(self.turn_step)
        if action == 3: # turn right
            self.turn_agent(-self.turn_step)
        if action == 4: # pick
            reward, done, info = self.pick()
            rewards += reward
        if action == 5: # drop
            reward, done, info = self.drop()
            rewards += reward
        # if action == 6:
        #     pass

        # done, info = self.move(self.curr_pos)
        #for i in range(self.load_steps):
        p.stepSimulation()
        self.load_camera()
        self.step_count += 1

        # if not done:
        #     if pick_or_drop == 0:
        #         reward, done, info = self.pick()
        #         if self.backpack[1] != -1:
        #             if self.prev_picked_color != self.backpack[-1]:
        #                 rewards += 3.0
        #                 self.prev_picked_color = self.backpack[-1] 
        #     else:
        #         reward, done, info = self.drop()
        #     rewards += reward

        self.pre_pos = self.curr_pos
        self.curr_pos = self.get_gtpose()
        
        self.current_dis_to_target = self.compute_configuration_distance()
        reward_configuration = self.pre_dis_to_target - self.current_dis_to_target
        rewards += reward_configuration
        self.pre_dis_to_target = self.current_dis_to_target
        state = self.get_state()
        # self.visualize() # for human gui, comment out in real
        info = {"current": info, "reward_config": reward_configuration, "picks": self.successful_picks}
        if self.step_count >= self.max_episode_steps:
            done = True
        return state, rewards, done, info

    def get_info(self, idx=-1):
        """
        Get reward, done, info after performing pick/drop
        input: idx=-1 (optional) only for drop action idx of object
        output: reward (only after drop), done, info
        """
        reward = 0
        done = False
        info = None
        if idx != -1:
            if self.near(self.objects[idx], self.objects[idx], self.object_init_pos[idx]):
                self.num_correct_drops_for_obj[idx] += 1
                info = "repeated"
                if self.num_correct_drops_for_obj[idx] == 1:
                    self.num_correct_drops += 1
                    reward = 10
                    info = "placed"
        if self.num_correct_drops == self.num_objects:
            done = True
            for i in range(self.num_objects):
                if i == self.backpack[1]: continue
                if not self.near(self.objects[i], self.objects[i], self.object_init_pos[i]):
                    done = False
                    self.num_correct_drops -=1

        if done:
            info = "succeeded"

        if self.step_count >= self.max_episode_steps:
            done = True
            if info != 'succeeded':
                info = 'failed'
            
        # if done:
        #     self.visualize() # for human gui, comment out in real

        return reward, done, info


    def near(self, ent0, ent1=None, pos2=None):
        """
        Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks
        """
        rad0 = 0.5
        if ent0 == self.robot or ent0[1] == self.cube_name:
            rad0 = math.sqrt(0.5)
        
        if ent1 is None:
            ent1 = self.robot

        rad1 = 0.5
        if ent1 == self.robot or ent1[1] == self.cube_name:
            rad1 = math.sqrt(0.5)

        pos1, _ = p.getBasePositionAndOrientation(ent0[0])

        if pos2 is None:
            pos2, _ = p.getBasePositionAndOrientation(ent1[0])
        

        dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return dist < rad0 + rad1 + 1.1 * 1.0

    def move_agent(self, fwd_dist, fwd_drift):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        # target = [pos[0]+math.cos(ori[2])*fwd_dist, pos[1]+math.sin(ori[2])*fwd_dist, pos[2]]
        yaw = p.getEulerFromQuaternion(ori)[2]
        target = [pos[0]+math.cos(yaw)*fwd_dist, pos[1]+math.sin(yaw)*fwd_dist, pos[2]]
        
        if self.collision_detection(target) != 1:
            return False

        p.resetBasePositionAndOrientation(self.robot, target, ori)
        return True

    #def move(self, target):
    def teleport(self, target):
        """
        Command agent to teleport to the target area, facing that direction
        input: target = (x,y)
        """
        self.step_count += 40
        done = False
        info = "tp agent"
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        tmp_ori = p.getEulerFromQuaternion(ori)
        curr = np.array([pos[0],pos[1],pos[2]])
        target = np.array(target)
        orig_target = target
        angle = self.compute_new_ori(curr, target)
        self.curr_pos[2] = angle
        dis = np.linalg.norm(curr-target)
        x = math.cos(angle)
        y = math.sin(angle)
        new_ori = [tmp_ori[0], tmp_ori[1], angle]
        new_ori = p.getQuaternionFromEuler(new_ori)
        dir = np.array([x,y,0])
        target = curr+dis*dir
        step = 0
        while True:
            success = self.collision_detection(target)
            if success == 1:
                break
            dis += -1.5 if success == -1 else -1.0
            if dis <= 0:
                dis = 0
                target = curr+dis*dir
                break
            target = curr+dis*dir
            step += 1
            if step == 5:
                target = curr+0*dir
                break

        if random.random() < 0.2:
            step = 0
            while True:
                next_pos_mean = [target[0], target[1]]
                cov = [[0.1*dis, 0], [0, 0.1*dis]]
                next_x, next_y= np.random.multivariate_normal(next_pos_mean, cov)
                next_pos = [next_x, next_y, 0.5]
                if self.collision_detection(next_pos) != 1:
                    step += 1
                    if step == 5: break
                    continue
                target = next_pos
                break

        if self.step_count >= self.max_episode_steps:
            done = True

        p.resetBasePositionAndOrientation(self.robot, target, new_ori)
        return done, info

    def compute_new_ori(self, curr, target):
        """
        This and the next function calculate how the robot should face before teleporting
        input: curr, target = (x,y)
        output: angle in radians depicting where robot faces next
        """
        curr = np.array([curr[0], curr[1]])
        target = np.array([target[0], target[1]])
        heading_tar = target-curr

        if heading_tar[0] == 0 and heading_tar[1] == 0:
            return self.transform_angle(0)
        initial = [0,1]
        unit_vector_2 = heading_tar / np.linalg.norm(heading_tar)
        dot_product = np.dot(initial, unit_vector_2)
        angle = np.arccos(dot_product)
        angle = self.transform_angle(angle) if heading_tar[0]>=0 else \
            self.transform_angle(-angle)

        if angle >= 2*math.pi: angle -= 2*math.pi
        
        return angle

    def transform_angle(self, theta):
        """
        input: theta = angle in radians
        output: angle in radians that depicts where robot faces next
        """
        if theta <= 0.0:
            return np.abs(theta)+np.pi/2
        return 5*np.pi/2-theta

    def collision_detection(self, target):
        """
        Checks whether the target coordinate is not colliding with other objects/walls or is outside the map.
        input: target = (x,y)
        output: Returns -1 on wall collision, 1 on success, or the object item in self.objects
        """
        x,y,_ = target
        # if x+math.sqrt(0.5) >= self.length or \
        #     x-math.sqrt(0.5) <= 0  or \
        #     y+math.sqrt(0.5) >= self.length or \
        #     y-math.sqrt(0.5) <= 0:
        if x+math.sqrt(0.6) >= self.length or \
            x-math.sqrt(0.6) <= 0  or \
            y+math.sqrt(0.6) >= self.length or \
            y-math.sqrt(0.6) <= 0:
            # print(f"({x},{y}) is outside the map.")
            return -1
        for i in range(self.num_objects):
            if i == self.backpack[1]: continue
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            diff = math.sqrt((pos[0]-x)**2 +
                        (pos[1]-y)**2)
            radius = 0.5
            if self.objects[i][1] == self.cube_name:
                radius = math.sqrt(0.5)
            if diff < math.sqrt(0.5) + radius:
                # print("Something in the way.")
                return (self.objects[i], i)
        # print("Successful.")
        return 1

    def turn_agent(self, turn_angle):
        turn_angle *= math.pi / 180
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        new_ori = p.getEulerFromQuaternion(ori)
        new_ori = [new_ori[0],new_ori[1],new_ori[2]]
        new_ori[2] += turn_angle
        new_ori = p.getQuaternionFromEuler(new_ori)
        p.resetBasePositionAndOrientation(self.robot,pos,new_ori)

    def turn(self, dir):
        """
        Command robot to turn a specific direction
        input: dir = 1 or -1
        """
        theta = 5.0/180 * math.pi # change later to gaussian
        pos_ori = p.getBasePositionAndOrientation(self.robot)
        new_ori = p.getEulerFromQuaternion(pos_ori[1])
        new_ori = [new_ori[0],new_ori[1],new_ori[2]]
        new_ori[2] += theta * dir
        new_ori = p.getQuaternionFromEuler(new_ori)
        p.resetBasePositionAndOrientation(self.robot,pos_ori[0],new_ori)

    def pick(self):
        """
        Robot grabs object in fpv and stores in virutal backpack
        output: reward, done, info from get_info()
        """
        reward = 0
        done = False
        info = "Pick failed bc nothing near"
        #self.step_count += 1
        if self.backpack[1] != -1:
            # print("Backpack is full. Please empty.")
            info = "Pick failed bc backpack is full"
            reward, done, new_info = self.get_info()
            if new_info != None: info = new_info
            return reward, done, info
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]
        x_delta = math.cos(yaw) * math.sqrt(0.5)
        y_delta = math.sin(yaw) * math.sqrt(0.5)
        grab_pos = [pos[0]+x_delta, pos[1]+y_delta, pos[2]]
        success = self.collision_detection(grab_pos)
        if success != 1 and success != -1:
            info = "Picked properly"
            robot = success[0]
            idx = success[1]
            p.removeBody(robot[0])
            self.backpack[0] = robot
            self.backpack[1] = idx
            self.successful_picks+=1
            # print("Agent picked up an object.")
        #else:
            # print("Backpack is empty.")
        reward, done, new_info = self.get_info()
        if new_info != None: info = new_info
        return reward, done, info

    def drop(self):
        """
        Robot drops object infront of it if no obstacles
        output: reward, done, info from get_info()
        """
        #self.step_count += 1
        reward = 0
        done = False
        info = "Failed drop bc backpack empty"
        if self.backpack[1] != -1:
            pos_ori = p.getBasePositionAndOrientation(self.robot)
            x,y,_ = pos_ori[0]
            yaw = p.getEulerFromQuaternion(pos_ori[1])[-1]
            x_yaw = math.cos(yaw)
            y_yaw = math.sin(yaw)
            x = x+x_yaw
            y = y+y_yaw
            success = self.collision_detection([x,y,0])
            if success == 1:
                # print("Dropping object.")
                info = "Dropped properly"
                pos = [x,y,self.spawn_height]
                ori = pos_ori[1]
                type = self.backpack[0][1]
                idx = self.backpack[1]
                if type == "cylinder":
                    cylinder_collision_id = p.createCollisionShape(p.GEOM_CYLINDER)
                    cylinder_visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, rgbaColor=self.color_pool_value[self.color_ind[idx]])
                    cylinder_id = p.createMultiBody(0, cylinder_collision_id, cylinder_visual_id, pos)
                    self.objects[idx] = (cylinder_id, "cylinder")
                elif type == "cone":
                    cone_collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.cone_name, meshScale=[0.5,0.5,0.5])
                    cone_visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.cone_name,
                                meshScale=[0.5,0.5,0.5], rgbaColor=self.color_pool_value[self.color_ind[idx]])
                    ori = [math.pi/2,0,0]
                    ori = p.getQuaternionFromEuler(ori)
                    cone_id = p.createMultiBody(0, cone_collision_id, cone_visual_id, pos, ori)
                    self.objects[idx] = (cone_id, "cone")
                else:
                    object_cubeorshpere_id = p.loadURDF(fileName=type, 
                                                            basePosition=pos,baseOrientation=ori)
                    p.changeVisualShape(object_cubeorshpere_id, -1, rgbaColor=self.color_pool_value[self.color_ind[idx]])
                    self.objects[idx] = (object_cubeorshpere_id, type)
                self.backpack = [(-1, ""),-1]
                reward, done, new_info = self.get_info(idx=idx)
                if new_info != None: info = new_info
            else:
                info = "Failed drop bc smth in the way"
        return reward, done, info

    def spawn_wall(self):
        """
        **FROM SPATIAL ACTION MAPS GITHUB**
        spawns the four surroundings walls
        """
        obstacle_color = (1, 1, 1, 1)
        obstacles = []
        for x, y, length, width in [
                (-self.width, 6, self.width, self.length+self.width),
                (self.length+self.width, 6, self.width, self.length+self.width),
                (6, -self.width, self.length+self.width, self.width),
                (6, self.length+self.width, self.length+self.width, self.width)
            ]:
            obstacles.append({'type': 'wall', 'position': (x, y), 'heading': 0, 'length': length, 'width': width})

        seen = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = dir_path + "/../resources/wall_texture/wall_checkerboard_"
        for obstacle in obstacles:
            obstacle_half_extents = [obstacle['length'] / 2, obstacle['width'] / 2, self.wall_height]
            obstacle_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
            obstacle_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=obstacle_color)
            obstacle_id = p.createMultiBody(
                0, obstacle_collision_shape_id, obstacle_visual_shape_id,
                [obstacle['position'][0], obstacle['position'][1], 0.5], p.getQuaternionFromEuler([0, 0, obstacle['heading']])
            )
            while True:
                id = random.randint(0,199)
                if id not in seen:
                    seen.append(id)
                    break
            x = p.loadTexture(filename + str(id) + ".png")
            p.changeVisualShape(obstacle_id, -1, textureUniqueId=x)

    def randomize_objs_pos(self):
        """
        Chooses random locations for the objects
        output: list size n for n objects with each item being (x,y)
        """
        lastpos = [self.init_pos]
        i = 0
        for i in range(self.num_objects):
            # x = random.uniform(math.sqrt(0.5),self.length-math.sqrt(0.5))
            # y = random.uniform(math.sqrt(0.5),self.length-math.sqrt(0.5))
            x = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
            y = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
            j = 0
            while j < i+self.num_robots:
                pos = lastpos[j]
                diff = math.sqrt((pos[0]-x)**2 +
                            (pos[1]-y)**2)
                if (diff <= 2*math.sqrt(0.5)):
                    # x = random.uniform(math.sqrt(0.5),self.length-math.sqrt(0.5))
                    # y = random.uniform(math.sqrt(0.5),self.length-math.sqrt(0.5))
                    x = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
                    y = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
                    j = -1
                j += 1
            pos = [x,y,self.spawn_height]
            lastpos.append(pos)
        return lastpos[1:]

    def spawn_objects(self, init=False):
        """
        spawns the objects within the walls and no collision
        input: init = False; if True, store the init values
        """
        obj_poses = self.randomize_objs_pos()
        if init: 
            self.object_init_pos = obj_poses

            self.color_pool_name = ["green", "blue", "red", "skyblue", "yellow", "purple"]
            self.color_pool_value = [[0,1,0,1],[0, 0, 1, 1],[1, 0, 0, 1],[0,1,1,1], [1,1,0,1], [1,0,1,1]]

            self.color_ind = np.random.choice(6, self.num_objects, replace=False)

            self.objs_types_index = []

            for i in range(self.num_objects):
                choice = random.randint(0,3)
                self.objs_types_index.append(choice)
             
                pos = obj_poses[i]
                color_name_of_single_obj = self.color_pool_name[self.color_ind[i]]
                color_value_of_single_obj = self.color_pool_value[self.color_ind[i]]

                # print(color_name_of_single_obj, choice)

                if choice == 0:
                    sphere_id = p.loadURDF(fileName=self.sphere_name, basePosition=pos)
                    p.changeVisualShape(sphere_id, -1, rgbaColor=color_value_of_single_obj)
                    self.objects.append((sphere_id, self.sphere_name))
                
                elif choice == 1:
                    yaw = random.uniform(0, 2*math.pi)
                    ori = [0, 0, yaw]
                    ori = p.getQuaternionFromEuler(ori)
                    cube_id = p.loadURDF(fileName=self.cube_name, basePosition=pos, baseOrientation=ori)
                    p.changeVisualShape(cube_id, -1, rgbaColor=color_value_of_single_obj)
                    self.objects.append((cube_id , self.cube_name))
                    
                elif choice == 2:                
                    cylinder_collision_id = p.createCollisionShape(p.GEOM_CYLINDER)
                    cylinder_visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, rgbaColor=color_value_of_single_obj)
                    cylinder_id = p.createMultiBody(0, cylinder_collision_id, cylinder_visual_id, pos)
                    self.objects.append((cylinder_id, "cylinder"))
                    
                else:
                    ori = [math.pi/2,0,0]
                    ori = p.getQuaternionFromEuler(ori)
                    cone_collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.cone_name, meshScale=[0.5,0.5,0.5])
                    cone_visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.cone_name,
                                meshScale=[0.5,0.5,0.5], rgbaColor=color_value_of_single_obj)
                    cone_id = p.createMultiBody(0, cone_collision_id, cone_visual_id, pos, ori)
                    self.objects.append((cone_id,"cone"))

        else:
            for i in range(self.num_objects):
                _, ori = p.getBasePositionAndOrientation(self.objects[i][0])
                if self.objects[i][1] == self.cube_name: 
                    yaw = random.uniform(0, 2*math.pi)
                    ori = [0, 0, yaw]
                    ori = p.getQuaternionFromEuler(ori)
                p.resetBasePositionAndOrientation(self.objects[i][0], obj_poses[i], ori)

    def backpack_onehot(self):
        onehot = [1,1,1]
        if self.backpack[1] != -1:
            onehot[self.backpack[1]] == 0
        return onehot

    def get_action_space(self):
        return BEV_PIXEL_WIDTH*BEV_PIXEL_WIDTH

    def seed(self, seed):
        """
        sets default seed based on rank
        input: seed = any integer
        """
        self.seed = seed
        random.seed(self.seed)

    def output_objects(self):
        """
        Output the list of objects; purely for multiprocessing debugging purposes
        """
        print("rank: ", self.seed, ", ", self.object_init_pos)

    def get_object_gtpose(self):
        poses = []
        for i in range(self.num_objects):
            if i == self.backpack[1]: 
                agent_pos = self.get_gtpose()[:2]
                poses.extend(agent_pos)
            else:
                gtpos = p.getBasePositionAndOrientation(self.objects[i][0])[0]
                poses.extend([gtpos[0], gtpos[1]])
        return poses

    def get_gtpose(self):
        """
        Return gt pose for lnet
        """
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        ori = p.getEulerFromQuaternion(ori)
        #return np.array([pos[0], pos[1], ori[2]])
        return [pos[0], pos[1], ori[2]]
    
    def close(self):
        """
        close the pybullet connection
        """
        p.disconnect(self.client)

class Box(gym.Wrapper):
    def __init__(self, isGUI):
        env = BoxWorld(isGUI)
        super().__init__(env)
