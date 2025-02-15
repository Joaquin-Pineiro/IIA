import numpy as np
from CoppeliaSocket import CoppeliaSocket

class Environment:
    def __init__(self, sim_measurement, obs_dim, act_dim, rwd_dim):
        '''
        Creates a 3D environment using CoppeliaSim, where an agent capable of choosing its joints' angles tries to find
        the requested destination.
        :param obs_dim: Numpy array's shape of the observed state
        :param act_dim: Numpy array's shape of the action
        :param dest_pos: Destination position that the agent should search
        '''
        self.name = "ComplexAgentSAC"
        self.sim_measurement = sim_measurement                      # Simulation only measurements
        self.obs_dim = obs_dim                                      # Observation dimension
        self.act_dim = act_dim                                      # Action dimension
        self.rwd_dim = rwd_dim                                      # Reward dimension
        self.__end_cond = 11                                        # End condition (radius in meters)
        self.__obs = np.zeros((1,self.obs_dim))                     # Observed state
        self.__coppelia = CoppeliaSocket(obs_dim+sim_measurement)   # Socket to the simulated environment

        #Parameters for forward velocity reward
        self.forward_velocity_reward = 0
        self.__target_velocity = 0.3 # m/s (In the future it could be a changing velocity)
        self.__vmax = 2
        self.__delta_vel = 0.6
        self.__vmin = -6

        self.__curvature_forward_vel = - 2* self.__vmax / (self.__delta_vel * self.__vmin)

        #Parameters for forward acceleration penalization
        self.forward_acc_penalty = 0
        self.__max_acc = 8 #m/s^2  (Acceleration at which the penalization is -1)

        #Parameters for lateral velocity penalization
        self.lateral_velocity_penalty = 0
        self.__vmin_lat = -2
        self.__curvature_lateral = 3
        
        #Parameters for orientation reward
        self.rotation_penalty = 0
        self.__vmin_rotation = -6
        self.__curvature_rotation = 6
        
        #Parameters for flat back reward
        self.flat_back_penalty = np.zeros(2)
        self.__vmin_back = -2
        self.__curvature_back = 2

        #Reward for not flipping over
        self.__not_flipping_reward = 0.5

    def reset(self):
        ''' Generates and returns a new observed state for the environment (outside of the termination condition) '''
        # Start position in (0,0) and orientation (in z axis)
        pos = np.zeros(2)
        # z_ang = 2*np.random.rand(1) - 1 #vector of 1 rand between -1 and 1, later multiplied by pi
        z_ang = np.array([0])
        
        # Join position and angle in one vector
        pos_angle = np.concatenate((pos,z_ang))

        # Reset the simulation environment and obtain the new state
        self.__obs = self.__coppelia.reset(pos_angle)
        return np.copy(self.__obs)

    def act(self, act):
        ''' Simulates the agent's action in the environment, computes and returns the environment's next state, the
        obtained reward and the termination condition status '''
        # Take the requested action in the simulation and obtain the new state
        next_obs = self.__coppelia.act(act.reshape(-1))
        # Compute the reward
        reward, end = self.compute_reward_and_end(self.__obs.reshape(1,-1), next_obs.reshape(1,-1))
        # Update the observed state
        self.__obs[:] = next_obs
        # Return the environment's next state, the obtained reward and the termination condition status
        return next_obs, reward, end

    def compute_reward_and_end(self, obs, next_obs):
        # Compute reward for every individual transition (state -> next_state)

            # Final distance to evaluate end condition
        #dist_fin = np.sqrt(np.sum(np.square(next_obs[:,0:2]), axis=1, keepdims=True))

            # Velocity vector from every state observed

        carrito = next_obs[:,0]
        carrito_vel = next_obs[:,1]
        carrito_acc = next_obs[:,2]
        carrito_force = next_obs[:,3]

        pendulo1 = np.arctan2(next_obs[:,5],next_obs[:,4])
        pendulo2 = np.arctan2(next_obs[:,7],next_obs[:,6]) 
        #pendulo3 = next_obs[:,6] * np.pi 

        pendulo1_vel = next_obs[:,8]
        pendulo2_vel = next_obs[:,9]
        #pendulo2_vel = next_obs[:,9]


        # Empty vectors to store reward and end flags for every transition
        reward, end = np.zeros((obs.shape[0], self.rwd_dim)), np.zeros((obs.shape[0], 1))

        for i in range(obs.shape[0]):

            
            #reward[i,0] = np.cos(pendulo1)+0.5

            # print("Pendulo1 = ", pendulo1*180/np.pi)
            # print("Pendulo2 = ", pendulo2*180/np.pi)

            # if (abs(pendulo1)>=(12*np.pi/180)) or (abs(pendulo2)>=(6*np.pi/180)) or (abs(pendulo3)>=(1*np.pi/180)):

            #     reward[i,0] = -2
            #     #print("ENTRE")


            # if abs(pendulo1)<=4*np.pi/180 and abs(pendulo2)<=0.5*np.pi/180 and abs(pendulo3)<=0.05*np.pi/180:

            #     reward[i,0] = +1


            # reward1 = -np.exp(5*np.abs(pendulo1))+2
            # reward2 = -np.exp(5*np.abs(pendulo2))+2

            # if reward1<=0 and reward2<=0:
            #     reward[i,0] = -reward1*reward2
            # else:
            #     reward[i,0] = reward1*reward2

            #reward[i,0] = np.cos(pendulo1)*2

            #reward[i,1] = np.cos(pendulo2)*4

            reward[i,0] = 3*np.exp(-1*abs(pendulo1))-2
            reward[i,1] = 3*np.exp(-1*abs(pendulo2))-2
            
            reward[i,2] = -0.5*pendulo1_vel**2

            reward[i,3] = -0.5*pendulo2_vel**2

            #reward[i,3] = -0.1*pendulo2_vel**2

            #reward[i,2] = -0.01*carrito_force**2

            reward[i,4] = -0.0001*carrito_acc**2

            reward[i,5] = -0.01*carrito_vel**2

            #print("Pendulo2 rwd",reward[i,1])







            if carrito >=2 or carrito <= -2:
                reward[i,6] = -1
            #else: 
             #   end[i] = False

            #if np.abs(pendulo1) >= 25*np.pi/180 or np.abs(pendulo2) >= 25*np.pi/180 or np.abs(pendulo3) >= 12*np.pi/180:
            #   end[i] = True 

            #print(pendulo1)

        return reward, end