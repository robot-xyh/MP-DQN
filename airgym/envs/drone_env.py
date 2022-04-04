#import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
import cv2
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.drone = airsim.MultirotorClient()
        client = airsim.VehicleClient()
        client.confirmConnection()
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            "kk":13,
        }
        client.simRunConsoleCommand("t.MaxFPS 10")
        """
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthVis , pixels_as_float=True,
            compress=False
        )
        """
        self.image_request_0_scene = airsim.ImageRequest(
            "0", airsim.ImageType.Scene , False,
            False
        )
        self.image_request_1_DV = airsim.ImageRequest(
            "1", airsim.ImageType.DepthVis , True,
            False
        )
        self.image_request_2_DV = airsim.ImageRequest(
            "2", airsim.ImageType.DepthVis , True,
            False
        )
        self.image_request_3_DV = airsim.ImageRequest(
            "3", airsim.ImageType.DepthVis , True,
            False
        )
        self.image_request_4_DV = airsim.ImageRequest(
            "4", airsim.ImageType.DepthVis , True,
            False
        )
        self._setup_flight()
        self.step_length = 3.14/180
        self.camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        time.sleep(0.01)
        self.drone.simSetCameraPose("0", self.camera_pose)
        self.o = np.array([[1.],[0.],[0.]])
        
        self.goal = np.array([
                        105,-10,-50
                        ])
        my_pos=self.drone.getMultirotorState().kinematics_estimated.position
        quad_pt = np.array(list((my_pos.x_val, my_pos.y_val, my_pos.z_val)))
        self.dis_goal = np.linalg.norm((quad_pt - self.goal))
        self.dist = self.dis_goal
        self.dist_prev = self.dist
        self.s_t=0.55
        self.prev_s_t = self.s_t
        
        """
        self.goal = np.array([
                        self.drone.simGetObjectPose("OrangeBall_Blueprint").position.x_val, 
                        self.drone.simGetObjectPose("OrangeBall_Blueprint").position.y_val,
                        self.drone.simGetObjectPose("OrangeBall_Blueprint").position.z_val
                        ])
        print(self.drone.simGetObjectPose("OrangeBall_Blueprint"))
        """
        self.win  = 0
        self.gimbal_win = 0
        self.mv_win = 0
        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "collision": None,
            "pose": None,
            "prev_pose": None,
            "dist": self.dist,
            "angle":self.s_t,
            "win": self.win,
            "gimbal_win":self.gimbal_win,
            "mv_win": self.mv_win
        }
        
        #self.episode_horizon = episode_horizon
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0
        self.step_num = 0
        self.error_count = 0
        self.det_error_count = 0
        self.distance = self.drone.getDistanceSensorData("Distance","Drone0").distance
        self.gim_step = 0
        self.mv_step = 0
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Tuple((spaces.Discrete(8),
                                        
                                          spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                                          spaces.Box(low=-1, high=1, shape=(3,),dtype=np.float32),
                                          spaces.Box(low=-1, high=1, shape=(3,),dtype=np.float32),
                                          spaces.Box(low=-1, high=1, shape=(3,),dtype=np.float32),
                                          spaces.Box(low=-1, high=1, shape=(3,),dtype=np.float32),
                                          spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                                          spaces.Box(low=-1, high=1, shape=(3,),dtype=np.float32),
                                          spaces.Box(low=-1, high=1, shape=(3,),dtype=np.float32)
                                          ))
        print(self.action_space,"self.action_space")    
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        import airsim
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.takeoffAsync().join()
        self.drone.hoverAsync().join()
        #self.drone.moveToPositionAsync(-0, 0,-50, 5).join()
        #self.drone.moveToPositionAsync(-100, 0,-50, 5).join()
        self.camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        time.sleep(0.01)
        self.drone.simSetCameraPose("0", self.camera_pose)
        
        self.camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion( -math.pi/9, 0,-math.pi/3))
        time.sleep(0.01)
        self.drone.simSetCameraPose("1", self.camera_pose)
        self.camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion( -math.pi/9, 0,math.pi/3))
        time.sleep(0.01)
        self.drone.simSetCameraPose("2", self.camera_pose)
        self.camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-math.pi/9, 0, 0))
        time.sleep(0.01)
        self.drone.simSetCameraPose("3", self.camera_pose)
        self.camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-math.pi/2, 0, 0))
        time.sleep(0.01)
        self.drone.simSetCameraPose("4", self.camera_pose)
        
        self.goal = np.array([
                        105,-10,-50
                        ])
        my_pos=self.drone.getMultirotorState().kinematics_estimated.position
        quad_pt = np.array(list((my_pos.x_val, my_pos.y_val, my_pos.z_val)))
        self.dis_goal = np.linalg.norm((quad_pt - self.goal))
        self.dist = self.dis_goal
        self.dist_prev = self.dist
        



    def _get_obs(self):
        import airsim
        from PIL import Image

        # Update our (internal) state.
        self.drone_state = self.drone.getMultirotorState()
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.drone_state.kinematics_estimated
        

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        self.state["collision"] = collision
        self.state["dist"] = self.dist
        self.state["win"] = self.win
        self.state["mv_win"] = self.mv_win
        self.state["gimbal_win"] = self.gimbal_win
        from PIL import Image
        response_rgbs = self.drone.simGetImages([self.image_request_0_scene])
        img_rgb_out = []
        for i, response_rgb in enumerate(response_rgbs):
            img1d_rgb = np.fromstring(response_rgb.image_data_uint8, dtype=np.uint8) # get numpy array
            if (img1d_rgb.shape[0])==110592:
                img_rgb = img1d_rgb.reshape(response_rgb.height, response_rgb.width, 3) # reshape array to 4 channel image array H X W X 3
                
                img_rgb = Image.fromarray(img_rgb)
                img_rgb = img_rgb.resize((84,84),Image.ANTIALIAS)
                #img_rgb = np.array(img_rgb.convert("L"))
                #print(img_rgb.shape)
                if i == 0 :
                    img_rgb = np.array(img_rgb.resize((84, 84)).convert("RGB"))
                    img_rgb_out.append( img_rgb)
                else:
                    img_rgb = np.array(img_rgb.resize((84, 84)).convert("L"))
                    img_rgb = img_rgb.reshape([84, 84, 1]).astype(np.uint8)
                    img_rgb_out.append(img_rgb )
                #img_rgb = img_rgb.reshape([84, 84, 3]).astype(np.float32)
                #img_rgb /= 255.0
                #print(img_rgb.shape,"-----------------------------")
                #airsim.write_png(os.path.normpath('aaaa' + str(i)+ '.png'), img_rgb)
            else:
                if i == 0 :
                    img_rgb_out.append( np.ones((84,84,3)).astype(np.uint8))
                else:
                    img_rgb_out.append( np.ones((84,84,1)).astype(np.uint8))
                    
        
        img_deep_out = []
        responses_deeps = self.drone.simGetImages([  self.image_request_1_DV,
                                                    self.image_request_2_DV,
                                                    self.image_request_3_DV,
                                                    self.image_request_4_DV
                                                ])
        for i, responses_deep in enumerate(responses_deeps):                                        
            img1d = np.array(responses_deep.image_data_float, dtype=np.float)
            if img1d.shape[0]== 36864:
                
                depth_img_in_meters = airsim.list_to_2d_float_array(responses_deep.image_data_float, responses_deep.width, responses_deep.height)
                depth_img_in_meters = depth_img_in_meters.reshape(responses_deep.height, responses_deep.width, 1)
                from PIL import Image
                MIN_DEPTH_METERS = 0
                MAX_DEPTH_METERS = 100
                
                
                from PIL import Image
                # Lerp 0..100m to 0..255 gray values
                depth_8bit_lerped = np.interp(depth_img_in_meters, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255))
                (depth_8bit_lerped) = (depth_8bit_lerped).reshape([144, 256])
                #print((depth_8bit_lerped).shape)
                
                image = Image.fromarray(np.uint8(depth_8bit_lerped))
                im_final = np.array(image.resize((84, 84)).convert("L"))
                depth_8bit_lerped = im_final.reshape([84, 84, 1])
                img_deep_out.append(depth_8bit_lerped)
                #print("deep right")
                #self.right_count +=1
                #airsim.write_png(os.path.normpath('right'+ "-"+str(i) + '.png'), img)
            else:
                #self.error_count +=1
                img = np.ones((84,84,1)).astype(np.uint8)
                img_deep_out.append(img)
                print("deep error")
                airsim.write_png(os.path.normpath('error'+  '.png'), img)
        
        
        #print(img_rgb_out[0].dtype)
        #print(img_deep_out[0].dtype)
        img_rgb_conca = np.concatenate((img_rgb_out[0],img_deep_out[0],img_deep_out[1],img_deep_out[2],img_deep_out[3]),axis = 2)
        #print (img_rgb_conca.shape)       
        return img_rgb_conca.reshape(7,84,84) 




    def step(self, action):
        self._set_actions(action)
        self._set_g_actions(action)
        # Get next observation, rewards, and dones..
        obs = self._get_obs()
        #print(obs,"obs")
        reward, done = self._get_rewards_and_dones()

        return obs, reward, done, self.state

    def reset(self):
        """Calls reset on the airSimDroneEnv."""
        import airsim
        self.episode_timesteps = 0
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        time.sleep(0.1)
        self.camera_pose=airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        self.drone.simSetCameraPose("0", self.camera_pose)
        self.camera_pose=airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        self.drone.simSetCameraPose("1", self.camera_pose)
        self.camera_pose=airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        self.drone.simSetCameraPose("2", self.camera_pose)
        self.camera_pose=airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        self.drone.simSetCameraPose("3", self.camera_pose)
        self.camera_pose=airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0))
        self.drone.simSetCameraPose("4", self.camera_pose)
        self._setup_flight()
        self.gim_step = 0
        self.mv_step = 0
        self.win = 0
        self.gimbal_win = 0
        self.mv_win = 0
        return self._get_obs()

    def interpret_action(self, action):
        g_quad_offset = (0,0,0)
        quad_offset = (0,0,0)
        if action[0] == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action[0] == 1:
            quad_offset = (0, self.step_length, 0)
        elif action[0] == 2:
            quad_offset = (0, 0, self.step_length)
        elif action[0] == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action[0] == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action[0] == 5:
            quad_offset = (0, 0, -self.step_length)
        elif action[0] == 6:
            quad_offset = (0, 0, 0)
        elif action[0] == 7: 
            g_quad_offset = g_quad_offset
        return quad_offset, g_quad_offset
    
    def _set_g_actions(self,action):
        """Sends a valid gym action to the AirSim drone."""
        # By default, do not break and hit the gas pedal (throttle=1).
        import airsim
        action = action
        #print("action[1]",action)
        _,g_quad_offset = self.interpret_action(action)
        u, v, w = airsim.utils.to_eularian_angles(self.drone.simGetCameraInfo("0").pose.orientation)
        u = g_quad_offset[0]*0.3+u
        if u > 0.:
            u = 0.
        if u < -3.14/2.5:
            u = -3.14/2.5
        #v = g_quad_offset[1]+v

        w = g_quad_offset[2]*0.3+w
        if v > 3.14/6.0:
            v = 3.14/6.0
        if v < -3.14/6.0:
            v = -3.14/6.0
        if w > 3.14/3.:
            w = 3.14/3.
        if w < -3.14/3.:
            w = -3.14/3.
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(u, v, w))
        self.drone.simSetCameraPose("0", camera_pose)
        
    def _set_actions(self, action):
        """Sends a valid gym action to the AirSim drone."""
        # By default, do not break and hit the gas pedal (throttle=1).
        action = action
        #print("action[0]",action)
        quad_offset,_ = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            #0,
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2]*0.4,
            #0,
            4,
        ).join()
    def _get_rewards_and_dones(self):
        
        thresh_dist = 2
        #pts = np.array([35, 33, -3])
        self.goal = np.array([
                        105,-10,-50
                        ])
        reward_d = -1
        self.mv_win = 0 
        self.gimbal_win = 0
        if self.state["collision"]:
            reward = -20
            print("00000000000000000000")
        else:
            quad_pt = np.array(list((self.state["position"].x_val, self.state["position"].y_val, self.state["position"].z_val)))
            self.disgoal = np.array([
                        105,-10,-50
                        ])
            self.dist = np.linalg.norm(quad_pt - self.disgoal)
            #print(self.state["position"])
            #print("self.dist",self.dist)
            #print("self.dist_prev",self.dist_prev)
            reward_diff = self.dist_prev-self.dist
            self.dist_prev = self.dist
            
            if self.dist < 10:
                self.mv_win = 1 
                reward_d = 20
            else:
                reward_speed = (
                    np.linalg.norm(
                        [
                            self.state["velocity"].x_val,
                            self.state["velocity"].y_val,
                            self.state["velocity"].z_val,
                        ]
                    )
                    - 0.5
                )

                reward_d = reward_diff

        
        
        done=0
        self.distance = self.drone.getDistanceSensorData("Distance","Drone0").distance
        
        import airsim
        u, v, w = airsim.utils.to_eularian_angles(self.drone.simGetCameraInfo("0").pose.orientation)       
        
        client = airsim.MultirotorClient()
        #client.confirmConnection()
        # set camera name and image type to request images and detections
        camera_name = "0"
        image_type = airsim.ImageType.Scene

        # set detection radius in [cm]
        client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 
        # add desired object name to detect in wild card/regex format
        client.simAddDetectionFilterMeshName(camera_name, image_type, "OrangeBall_Blueprint")
        rawImage = self.drone.simGetImage(camera_name, image_type)
        if not rawImage:
            print("+++++++++++++++++++++++++++++++++++++")
        #img1d = np.fromstring(rawImage.image_data_uint8, dtype=np.uint8)
        if (len(rawImage)) > 100:
            png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
            #cv2.imshow("AirSim", png)
            cylinders = client.simGetDetections(camera_name, image_type)
            self.s_t =0.55
            if cylinders:
                for cylinder in cylinders:
                    s = pprint.pformat(cylinder)
                    #print("Cylinder: %s" % s)

                    cv2.rectangle(png,(int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val)),(int(cylinder.box2D.max.x_val),int(cylinder.box2D.max.y_val)),(255,0,0),2)
                    cv2.putText(png, cylinder.name, (int(cylinder.box2D.min.x_val),int(cylinder.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12))
                    x_center = (cylinder.box2D.min.x_val + cylinder.box2D.max.x_val)/2
                    y_center = (cylinder.box2D.min.y_val + cylinder.box2D.max.y_val)/2
                    #print(x_center,y_center)
                    x_center = (x_center - png.shape[1]/2)/(png.shape[1])
                    y_center = (y_center - png.shape[0]/2)/(png.shape[0])
                    self.s_t = (x_center**2 + y_center**2)
                    self.xlength = cylinder.box2D.max.x_val-cylinder.box2D.min.x_val  
                    self.ylength = cylinder.box2D.max.y_val-cylinder.box2D.min.y_val 
            #cv2.waitKey(1)        
            #print("-------------",s_t)
        else:
            self.s_t = 0.55
            self.det_error_count +=1
            print("self.det_error_count",self.det_error_count)
        
        self.state["angle"] = self.s_t
    
        if self.dist <=10 :
        
            if self.s_t < 0.1:
                reward_s_t = 20 
                self.gimbal_win = 1
            else:
                reward_s_t =  self.prev_s_t - self.s_t
                self.gimbal_win = 0
            self.prev_s_t = self.s_t
        else :
            reward_s_t = self.prev_s_t - self.s_t
            self.prev_s_t = self.s_t
###########################################        

        self.distance = self.drone.getDistanceSensorData("Distance","Drone0").distance
        

        reward = reward_s_t + reward_d
        
   
        
        if self.distance < 0.2:
            reward = -20
        if self.state["collision"]:
            reward = -20
        if self.state["position"].x_val<-20 or self.state["position"].x_val > 150:
            reward = -20
        if self.state["position"].y_val< -100 or self.state["position"].y_val > 100:
            reward = -20
        if self.state["position"].z_val<-120:  
            reward = -20
        #print(reward,self.dist,reward_d)
        if reward <= -19:
            done = 1
        if reward >= 39:
            done = 1
            
        
        return reward,  done

