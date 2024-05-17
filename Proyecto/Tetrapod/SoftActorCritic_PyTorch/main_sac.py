import os
import numpy as np
import torch
from SAC import SAC_Agent
from EnvironmentTetrapodVelocidad import Environment
from TrainHistory import TrainHistory

import multiprocessing
import queue
import pyqtgraph as pg

import sys

from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsRectItem
from PyQt5 import QtCore

import atexit   #For saving before termination

data_type = np.float64

# Handler to run if the python process is terminated with keyboard interrupt, or if the socket doesn't respond (simulator or the real robot)
# IMPORTANT:   -If you want to intentionally interrupt the python process, do so only when the agent is interacting with the environment and not when it is learning
#               otherwise the networks will be saved in an incomplete state.
#              -If intentionally interrupted, the script takes time to save the training history, so don't close the console window right away
def exit_handler(agent, train_history):
    train_history.episode -= 1
    agent.save_models()
    agent.replay_buffer.save(train_history.episode)

    train_history.save()


def SAC_Agent_Training(q):

    env = Environment(sim_measurement=0, obs_dim=10, act_dim=1, rwd_dim=7)

    #The 24 values from coppelia, in order:
        #---- The first 7 not seen by the agent but used in the reward (simulation only measurements)--------
        #position (x,y,z)
        #Mean forward and lateral velocities with the reference frame of the agent
        #Maximum absolute forward acceleration measured during the previous step with the reference frame of the agent
        #yaw of the agent's body
        
        #---- The last 17 seen by the agent in the state vector (observation dimension)------------------------------
        #yaw of the agent's body
        #pitch and roll of the agent's body
        #pitch and roll angular velocities of the agent's body
        #the 12 joint angles

    load_agent = True
    test_agent = True
    load_replay_buffer_and_history = False   #if test_agent == True, only the train history is loaded (the replay_buffer is not used)
    
    episodes = 20000
    episode_steps = 700 #Maximum steps allowed per episode
    save_period = 500

    #Preference vector maximum and minimum values
    pref_max_vector = np.array([1.5,1.5,1,1,1,1,1])
    pref_min_vector = np.array([0.7, 0.7,0.0,0.0,0.0,0.0,0.0])
    pref_dim = pref_max_vector.size

    agent = SAC_Agent('Cuadruped', env, pref_max_vector, pref_min_vector, replay_buffer_size=1000000)
    
    agent.replay_batch_size = 1000

    agent.update_Q = 1  # The Q function is updated every episode
    agent.update_P = 1  # The policy is updated every 1 episode

    if load_agent:
        agent.load_models()

    ep_obs = np.zeros((episode_steps+1, env.obs_dim+env.sim_measurement), dtype=data_type)        # Episode's observed states
    ep_act = np.zeros((episode_steps, env.act_dim), dtype=data_type)            # Episode's actions
    ep_rwd = np.zeros((episode_steps, env.rwd_dim), dtype=data_type)            # Episode's rewards
    train_history = TrainHistory(max_episodes=episodes)

    if load_replay_buffer_and_history:
        train_history.load()

        if test_agent == False:
            agent.replay_buffer.load(train_history.episode)

        train_history.episode = train_history.episode + 1

    # Set the exit_handler only when training
    if test_agent == False: atexit.register(exit_handler, agent, train_history)

    while train_history.episode <= episodes:

        print("Episode: ", train_history.episode)

        ep_obs[0], done_flag = env.reset(), False

        # Testing
        if test_agent:
            #Use the user input preference for the test: [vel_forward, acceleration, vel_lateral, orientation, flat_back] all values [0;pref_max_value)
            pref = np.array([[1,1,1,1,1,1,1]])
            print("Preference vector: ", pref)
            for step in range(episode_steps):
                # Decide action based on present observed state (taking only the mean)
                ep_act[step] = agent.choose_action(ep_obs[step][env.sim_measurement:], pref, random=False)  #The agent doesn't receive the position and target direction although it is on the ep_obs vector for plotting reasons

                # Act in the environment
                ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])

                if done_flag: break

            ep_len = step + 1

        # Training
        else:
            #Generate random preference for the episode
            pref = np.random.random_sample((1,pref_dim)) * (pref_max_vector-pref_min_vector) + pref_min_vector
            print("Preference vector: ", pref)
            for step in range(episode_steps):
                # Decide action based on present observed state (random action with mean and std)
                ep_act[step] = agent.choose_action(ep_obs[step][env.sim_measurement:], pref)

                # Act in the environment
                ep_obs[step+1], ep_rwd[step], done_flag = env.act(ep_act[step])

                # Store in replay buffer
                agent.remember(ep_obs[step][env.sim_measurement:], ep_act[step], ep_rwd[step], ep_obs[step+1][env.sim_measurement:], done_flag)

                # End episode on termination condition
                if done_flag: break

            ep_len = step + 1

        # Compute total reward from partial rewards and preference of the episode
        tot_rwd = np.sum(pref * ep_rwd[:,:], axis=1)
        
        if test_agent:
            # Send the information for plotting in the other process through a Queue
            q.put(( test_agent, ep_obs[0:ep_len+1], tot_rwd[0:ep_len+1], ep_rwd[0:ep_len+1],  ep_act[0:ep_len+1] ))

        else:
            ######## Compute the real and expected returns and the root mean square error ##########
            # Real return:
            # Auxiliary array for computing return without overwriting tot_rwd
            aux_ret = np.copy(tot_rwd)
            pref_tensor = torch.tensor(pref, dtype=torch.float64).to(agent.P_net.device)

            # If the episode ended because the agent reached the maximum steps allowed, the rest of the return is estimated with the Q function
            # Using the last state, and last action that the policy would have chosen in that state
            if not done_flag:
                last_state = torch.tensor(np.expand_dims(ep_obs[step+1][env.sim_measurement:], axis=0), dtype=torch.float64).to(agent.P_net.device)
                last_action = agent.choose_action(ep_obs[step+1][env.sim_measurement:], pref, random=not(test_agent), tensor=True)
                aux_ret[step] += agent.discount_factor * agent.minimal_Q(last_state, last_action, pref_tensor).detach().cpu().numpy().reshape(-1)

            for i in range(ep_len-2, -1, -1): aux_ret[i] = aux_ret[i] + agent.discount_factor * aux_ret[i+1]
            train_history.ep_ret[train_history.episode, 0] = aux_ret[0]

            # Expected return at the start of the episode:
            initial_state = torch.tensor(np.expand_dims(ep_obs[0][env.sim_measurement:], axis=0), dtype=torch.float64).to(agent.P_net.device)
            initial_action = torch.tensor(np.expand_dims(ep_act[0], axis=0), dtype=torch.float64).to(agent.P_net.device)
            train_history.ep_ret[train_history.episode, 1] = agent.minimal_Q(initial_state, initial_action, pref_tensor).detach().cpu().numpy().reshape(-1)

            # Root mean square error
            train_history.ep_ret[train_history.episode, 2] = np.sqrt(np.square(train_history.ep_ret[train_history.episode,0] - train_history.ep_ret[train_history.episode, 1]))

            ######### Train the agent with batch_size samples for every step made in the episode ##########
            for step in range(ep_len):
                agent.learn(step)

            # Store the results of the episodes for plotting and printing on the console
            train_history.ep_loss[train_history.episode, 0] = agent.Q_loss.item()
            train_history.ep_loss[train_history.episode, 1] = agent.P_loss.item()
            train_history.ep_alpha[train_history.episode] = agent.log_alpha.exp().item()
            train_history.ep_entropy[train_history.episode] = agent.entropy.item()
            train_history.ep_std[train_history.episode] = agent.std.item()
            
            print("Replay_Buffer_counter: ", agent.replay_buffer.mem_counter)
            print("Q_loss: ", train_history.ep_loss[train_history.episode, 0])
            print("P_loss: ", train_history.ep_loss[train_history.episode, 1])
            print("Alpha: ", train_history.ep_alpha[train_history.episode])
            print("Policy's Entropy: ", train_history.ep_entropy[train_history.episode])

            # Send the information for plotting in the other process through a Queue
            q.put(( test_agent, ep_obs[0:ep_len+1], tot_rwd[0:ep_len+1], ep_rwd[0:ep_len+1],  ep_act[0:ep_len+1], \
                    train_history.episode, train_history.ep_ret[0:train_history.episode+1], train_history.ep_loss[0:train_history.episode+1], \
                    train_history.ep_alpha[0:train_history.episode+1], train_history.ep_entropy[0:train_history.episode+1], train_history.ep_std[0:train_history.episode+1]))            
        
            # Save the progress every save_period episodes, unless its being tested
            if train_history.episode % save_period == 0 and train_history.episode != 0:
                agent.save_models()
                agent.replay_buffer.save(train_history.episode)
                
                train_history.save()
        print("------------------------------------------")
        train_history.episode += 1

#Range of the joints for plotting the denormalized joint angles
pendulo1_min, pendulo1_max = 0.0, 360.0
pendulo1_mean = (pendulo1_min + pendulo1_max)/2
pendulo1_range = (pendulo1_max - pendulo1_min)/2

pendulo2_min, pendulo2_max = 0.0, 360.0
pendulo2_mean = (pendulo2_min + pendulo2_max)/2
pendulo2_range = (pendulo2_max - pendulo2_min)/2

def updatePlot():   
    global q, curve_Pendulo1,curve_Pendulo2, curve_Reward,  curve_pendulo1_rwd, curve_pendulo2_rwd,curve_pendulo1_vel_rwd, curve_pendulo2_vel_rwd, curve_force_rwd, curve_acc_rwd,curve_vel_rwd, \
        curve_P_Loss, curve_Q_Loss, curve_Real_Return, curve_Predicted_Return, curve_Return_Error, curve_Alpha, curve_Entropy, curve_Std
        
    # print('Thread ={}          Function = updatePlot()'.format(threading.currentThread().getName()))
    try:  
        results=q.get_nowait()

        test_agent = results[0]

        pendulo1_angle = np.arctan2(results[1][:,5],results[1][:,4]) * 180/np.pi
        #pendulo2_angle = results[1][:,4] *180 

        state_linspace = np.arange(0,len(pendulo1_angle), 1, dtype=int)
        next_state_linspace = np.arange(1,len(pendulo1_angle), 1, dtype=int)

        ####Inclination update

        curve_Pendulo1.setData(state_linspace, pendulo1_angle)
        #curve_Pendulo2.setData(state_linspace, pendulo2_angle)

        
        ####Rewards update
        total_rwd = results[2]
        pendulo1_rwd = results[3][:,0]
        pendulo2_rwd = results[3][:,1]
        pendulo1_vel_rwd = results[3][:,2]
        pendulo2_vel_rwd = results[3][:,3]
        #force_rwd = results[3][:,2]
        acc_rwd = results[3][:,4]
        vel_rwd = results[3][:,5]
              

        rwd_linspace = np.arange(1,len(total_rwd)+1, 1, dtype=int)   #Because there is no reward in the first state (step 0)

        curve_Reward.setData(rwd_linspace, total_rwd)
        curve_pendulo1_rwd.setData(rwd_linspace, pendulo1_rwd)
        curve_pendulo2_rwd.setData(rwd_linspace, pendulo2_rwd)
        curve_pendulo1_vel_rwd.setData(rwd_linspace, pendulo1_vel_rwd)
        curve_pendulo2_vel_rwd.setData(rwd_linspace, pendulo2_vel_rwd)
        #curve_force_rwd.setData(rwd_linspace,force_rwd)
        curve_acc_rwd.setData(rwd_linspace,acc_rwd)
        curve_vel_rwd.setData(rwd_linspace,vel_rwd)


        if test_agent == False:
            last_episode = results[5]
            episode_linspace = np.arange(0,last_episode+1,1,dtype=int)

            ####Returns update
            Real_Return_data = results[6][:,0]
            Predicted_Return_data = results[6][:,1]

            
            curve_Real_Return.setData(episode_linspace,Real_Return_data)
            curve_Predicted_Return.setData(episode_linspace, Predicted_Return_data)

            ####Returns error update
            Return_loss_data = results[6][:,2]

            curve_Return_Error.setData(episode_linspace,Return_loss_data)

            ####Qloss update
            Q_loss_data = results[7][:,0]

            curve_Q_Loss.setData(episode_linspace,Q_loss_data)

            ####Ploss update
            P_loss_data = results[7][:,1]

            curve_P_Loss.setData(episode_linspace,P_loss_data)

            ####Alpha update
            Alpha_data = results[8]
            curve_Alpha.setData(episode_linspace,Alpha_data)
            
            ####Entropy update
            Entropy_data = results[9]

            curve_Entropy.setData(episode_linspace, Entropy_data)

            ####Standard deviation update
            Std_data = results[10]

            curve_Std.setData(episode_linspace, Std_data)

    except queue.Empty:
        #print("Empty Queue")
        pass

if __name__ == '__main__':
    # print('Thread ={}          Function = main()'.format(threading.currentThread().getName()))
    app = QApplication(sys.argv)

    #Create a queue to share data between processes
    q = multiprocessing.Queue()

    #Create and start the SAC_Agent_Training process
    SAC_process=multiprocessing.Process(None,SAC_Agent_Training,args=(q,))
    SAC_process.start()

    # Create window
    
    grid_layout = pg.GraphicsLayoutWidget(title="Cuadruped - Training information")
    grid_layout.resize(1200,800)
    
    pg.setConfigOptions(antialias=True)

############################################### PLOTS #####################################################################

    
    ####Pendulo 1/2 plot
    plot_Inclination = grid_layout.addPlot(title="Pendulo1 & Pendulo2", row=0, col=3)
    plot_Inclination.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Inclination.showGrid(x=True, y=True)

    curve_Pendulo1 = plot_Inclination.plot(pen=(0,255,255), name='Pendulo1')
    curve_Pendulo2 = plot_Inclination.plot(pen=(255,0,255), name='Pendulo2')
    

    
    ####Rewards plot
    plot_Rewards = grid_layout.addPlot(title="Total and individual Rewards", row=0, col=5)
    plot_Rewards.addLegend(offset=(1, 1), verSpacing=-1, horSpacing = 20, labelTextSize = '7pt', colCount=3)
    plot_Rewards.showGrid(x=True, y=True)

    curve_Reward = plot_Rewards.plot(pen=(255,255,0), name='Total')
    curve_pendulo1_rwd = plot_Rewards.plot(pen=(0,255,0), name='pendulo1')
    curve_pendulo2_rwd = plot_Rewards.plot(pen=(0,255,0), name='pendulo2')
    curve_pendulo1_vel_rwd = plot_Rewards.plot(pen=(0,0,255), name='pendulo1_vel')
    curve_pendulo2_vel_rwd = plot_Rewards.plot(pen=(0,255,0), name='pendulo2_vel')
    curve_force_rwd = plot_Rewards.plot(pen=(0,255,255), name='force')
    curve_acc_rwd = plot_Rewards.plot(pen=(255,255,255), name='acc')
    curve_vel_rwd = plot_Rewards.plot(pen=(255,0,255), name='vel')



    ####Returns plot
    plot_Returns = grid_layout.addPlot(title="Real Return vs Predicted Return", row=2, col=0)
    plot_Returns.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Returns.showGrid(x=True, y=True)

    curve_Real_Return = plot_Returns.plot(pen=(255,0,0), name='Real')
    curve_Predicted_Return = plot_Returns.plot(pen=(0,255,0), name='Predicted')


    ####ReturnError plot
    plot_Return_Error = grid_layout.addPlot(title="RMSD of Real and Predicted Return", row=2, col=1)
    plot_Return_Error.showGrid(x=True, y=True)

    curve_Return_Error = plot_Return_Error.plot(pen=(182,102,247))


    ####Qloss plot
    plot_Q_Loss = grid_layout.addPlot(title="State-Action Value Loss", row=2, col=2)
    plot_Q_Loss.showGrid(x=True, y=True)

    curve_Q_Loss = plot_Q_Loss.plot(pen=(0,255,0))


    ####Ploss plot
    plot_P_Loss = grid_layout.addPlot(title="Policy Loss", row=2, col=3)
    plot_P_Loss.showGrid(x=True, y=True)

    curve_P_Loss = plot_P_Loss.plot(pen=(0,128,255))


    ####Alpha plot
    plot_Alpha = grid_layout.addPlot(title="Alpha (Entropy Regularization Coefficient)", row=2, col=4)
    plot_Alpha.showGrid(x=True, y=True)

    curve_Alpha = plot_Alpha.plot(pen=(255,150,45))


    ####Entropy and Standard deviation plot
    plot_Entropy_Std = grid_layout.addPlot(title="Policy's Entropy and Standard deviation", row=2, col=5)
    plot_Entropy_Std.addLegend(offset=(1, 1), verSpacing=-1)
    plot_Entropy_Std.showGrid(x=True, y=True)

    curve_Entropy = plot_Entropy_Std.plot(pen=(0,255,255), name='Entropy')
    curve_Std = plot_Entropy_Std.plot(pen=(255,0,0), name='Std')

####################################################################################################################
    
    #Force Grid minimum size
    grid_layout.ci.layout.setColumnMinimumWidth(0,300)
    grid_layout.ci.layout.setColumnMinimumWidth(1,300)
    grid_layout.ci.layout.setColumnMinimumWidth(2,300)
    grid_layout.ci.layout.setColumnMinimumWidth(3,300)
    grid_layout.ci.layout.setColumnMinimumWidth(4,300)
    grid_layout.ci.layout.setColumnMinimumWidth(5,300)
    grid_layout.ci.layout.setRowMinimumHeight(0,315)
    grid_layout.ci.layout.setRowMinimumHeight(1,315)
    grid_layout.ci.layout.setRowMinimumHeight(2,315)
    grid_layout.ci.layout.setHorizontalSpacing(5)
    grid_layout.ci.layout.setVerticalSpacing(0)

    #Timer to update plots every 1 second (if there is new data) in another thread
    timer = QtCore.QTimer()
    timer.timeout.connect(updatePlot)
    timer.start(1000)
    
    
    grid_layout.show()

    status = app.exec_()
    sys.exit(status)