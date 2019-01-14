#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append('../')

import skimage as skimage
from skimage import transform, color, exposure
from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time
import csv
import os
import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop
from keras import backend as K
import keras
import pandas as pd
from vizdoom import DoomGame, ScreenResolution
from vizdoom import *
import itertools as it
from time import sleep
import tensorflow as tf

from networks import Networks
# from segmentation_image import *

#to check if keras is using GPU

class DFPAgent:

    def __init__(self, state_size, measurement_size, action_size, timesteps):

        # get size of state, measurement, action, and timestep
        self.state_size = state_size
        self.measurement_size = measurement_size
        self.action_size = action_size
        self.timesteps = timesteps

        # these is hyper parameters for the DFP
        self.gamma = 0.99
        self.learning_rate = 0.00001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 64
        self.observe = 50000 #2000
        self.explore = 200000
        self.frame_per_action = 4
        self.timestep_per_train = 5 #5 # Number of timesteps between training interval

        # experience replay buffer
        self.memory = deque()
        self.max_memory = 20000

        # create model
        self.model = None

        # Performance Statistics
        self.stats_window_size= 5 # window size for computing rolling statistics
        self.mavg_score = [] # Moving Average of Survival Time
        self.var_score = [] # Variance of Survival Time

    def get_action(self, state, measurement, goal, inference_goal):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            #print("----------Random Action----------")
            action_idx = random.randrange(self.action_size)
        else:
            measurement = np.expand_dims(measurement, axis=0)
            goal = np.expand_dims(goal, axis=0)
            f = self.model.predict([state, measurement, goal]) # [1x6, 1x6, 1x6]
            f_pred = np.vstack(f) # 3x6
            obj = np.sum(np.multiply(f_pred, inference_goal), axis=1) # num_action

            action_idx = np.argmax(obj)
        return action_idx

    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, m_t, is_terminated):
        self.memory.append((s_t, action_idx, r_t, s_t1, m_t, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self, goal):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        rand_indices = np.random.choice(len(self.memory)-(self.timesteps[-1]+1), self.batch_size)

        state_input = np.zeros(((batch_size,) + self.state_size)) # Shape batch_size, img_rows, img_cols, img_channels
        measurement_input = np.zeros((batch_size, self.measurement_size))
        goal_input = np.tile(goal, (batch_size, 1))
        f_action_target = np.zeros((batch_size, (self.measurement_size * len(self.timesteps))))
        action = []

        for i, idx in enumerate(rand_indices):
            future_measurements = []
            last_offset = 0
            done = False
            for j in range(self.timesteps[-1]+1):
                if not self.memory[idx+j][5]: # if episode is not finished
                    if j in self.timesteps: # 1,2,4,8,16,32
                        if not done:
                            future_measurements += list( (self.memory[idx+j][4] - self.memory[idx][4]) )
                            last_offset = j
                        else:
                            future_measurements += list( (self.memory[idx+last_offset][4] - self.memory[idx][4]) )
                else:
                    done = True
                    if j in self.timesteps: # 1,2,4,8,16,32
                        future_measurements += list( (self.memory[idx+last_offset][4] - self.memory[idx][4]) )
            f_action_target[i,:] = np.array(future_measurements)
            state_input[i,:,:,:] = self.memory[idx][0][0,:,:,:]
            measurement_input[i,:] = self.memory[idx][4]
            action.append(self.memory[idx][1])

        f_target = self.model.predict([state_input, measurement_input, goal_input]) # Shape [32x18,32x18,32x18]

        for i in range(self.batch_size):
            f_target[action[i]][i,:] = f_action_target[i]

        loss = self.model.train_on_batch([state_input, measurement_input, goal_input], f_target)

        return loss

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name, overwrite=True)


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
    img = skimage.transform.resize(img,size)
    img = skimage.color.rgb2gray(img)

    return img

################################################################
#FOR COMPUTATION OF DEPTH MAP

from depth_map import *

################################################################""
#python3 dfp_extended_measures.py test 1 1

import argparse
import sys

if __name__ == '__main__':

    title = sys.argv[1]
    n_measures = int(sys.argv[2])  # number of measurements
    more_perception = int(sys.argv[3])
    test_phase = int(sys.argv[4])
    d2_environment = int(sys.argv[5])
    random_goal = int(sys.argv[6])

    sess = tf.Session()
    sess2 = tf.Session()
    try:
        sess.close()
    except NameError:
        pass
        try:
            sess2.close()
        except NameError:
            pass

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) #session depth map
    input_node, net = init_depth_map(sess)


    # Avoid Tensorflow eats up GPU memory
    config2 = tf.ConfigProto()
    config2.gpu_options.allow_growth = True
    sess2 = tf.Session(config=config2)
    K.set_session(sess2)

    game = DoomGame()

    if d2_environment:
        game.load_config("../vizdoom/scenarios/health_gathering_supreme.cfg")
    else:
        game.load_config("../vizdoom/scenarios/health_gathering.cfg")

    # TODO : Change amo/frags values when dealing with D3
    amo = 0
    frags = 0
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [Health]
    prev_misc = misc

    action_size = game.get_available_buttons_size() # [Turn Left, Turn Right, Move Forward]
    measurement_size = n_measures # [Health, Medkit, Poison]
    #timesteps = [1, 2, 4, 8, 16, 32, 64] 
    #timesteps = [1, 2, 4, 8, 16]
    timesteps = [1, 2, 4, 8, 16, 32]
    goal_size = measurement_size * len(timesteps)

    img_rows , img_cols = 84, 84
    # Convert image into Black and white
    if more_perception:
        img_channels = 2 # We stack 1 frame (then we will put 2 other channels: depth map and segmented image)
    else:
        img_channels = 1

    state_size = (img_rows, img_cols, img_channels)
    agent = DFPAgent(state_size, measurement_size, action_size, timesteps)

    agent.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps),
                                       agent.learning_rate)

    if d2_environment:
       agent.observe = 50000
       agent.explore = 200000
       tend = 240000
    else:
       agent.observe = 2000
       agent.explore = 50000
       tend = 60000


    if test_phase:
        print("Loading agent's weights for Test session...")
        agent.epsilon = 0
        agent.load_model('../../experiments/'+title+'/model/DFP.h5')
        agent.tend = 50000

    x_t = game_state.screen_buffer  # 480 x 640

    if more_perception:

        ############################################
        #COMPUTE DEPTH MAP
        img0 = np.rollaxis(x_t, 0, 3)
        npimg = np.round(255 * img0)
        img = Image.fromarray(npimg, 'RGB')
        depth_t = predict_depth_map(img, sess, input_node, net)[0, :, :, 0]
        # depth_t = predict_segmentation(img)

        ############################################
        # PROCESS IMAGE X_T (RESIZE AND TO GREYSCALE)
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))

        ############################################
        #PROCESS_IMAGE DEPTH_T (RESIZE)

        depth_t = transform.resize(depth_t, (img_rows, img_cols))
        depth_t = (depth_t - np.min(depth_t))/(np.max(depth_t)-np.min(depth_t))
        ############################################

        s_t = np.zeros((img_rows, img_cols,2))
        s_t[:,:,0] = x_t # It becomes 64x64x2
        s_t[:,:,1] = depth_t
        s_t = np.expand_dims(s_t, axis=0) # 1x64x64x2

    else:
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.expand_dims(x_t, axis=2) # It becomes 64x64x1
        s_t = np.expand_dims(s_t, axis=0) # 1x64x64x1


    # Number of medkit pickup as measurement
    medkit = 0

    # Number of poison pickup as measurement
    poison = 0

    # Initial normalized measurements
    assert(n_measures in [1,3])
    if n_measures==3:
        m_t = np.array([misc[0]/30.0, medkit/10.0, poison])
    elif n_measures==1:
        m_t = np.array([misc[0] / 30.0])

    # Goal
    if n_measures == 3:
        goal = np.array([1.0, 1.0, -1.0] * len(timesteps))
    elif n_measures==1:
        goal = np.array([1.0] * len(timesteps))

    # Goal for Inference (Can change during test-time)
    inference_goal = goal

    is_terminated = game.is_episode_finished()

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0 # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics
    life_buffer = []

    if not os.path.exists('../../experiments/'+title):
        os.mkdir('../../experiments/'+title)
    if not os.path.exists('../../experiments/'+title+'/model'):
        os.mkdir('../../experiments/'+title+'/model')
    if not os.path.exists('../../experiments/'+title+'/logs'):
        os.mkdir('../../experiments/'+title+'/logs')
    if not os.path.exists('../../experiments/'+title+'/statistics'):
        os.mkdir('../../experiments/'+title+'/statistics')

    csv_file = pd.DataFrame(columns=['Time', 'State', 'Epsilon', 'Action',
                                     'Reward', 'Medkit', 'Poison', 'Frags',
                                     'Amo', 'Max Life', 'Life', 'Mean Score',
                                     'Var Score', 'Loss'])
    csv_file.to_csv('../../experiments/' + title + '/logs/' + 'results.csv', sep=',', index=False)
    if test_phase:
        csv_file.to_csv('../../experiments/' + title + '/logs/' + 'results_test.csv', sep=',', index=False)
    if random_goal:
        inference_goal = goal = np.array(list(np.random.uniform(-1, 1, n_measures)) * len(timesteps))
    while not game.is_episode_finished():
        loss = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx  = agent.get_action(s_t, m_t, goal, inference_goal)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        r_t = game.get_last_reward()

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
            life_buffer.append(life)
            print ("Episode Finish ", misc)
            game.new_episode()
            if random_goal:
                inference_goal = goal = np.array(list(np.random.uniform(-1, 1, n_measures)) * len(timesteps))
            game_state = game.get_state()
            misc = game_state.game_variables
            x_t1 = game_state.screen_buffer

        x_t1 = game_state.screen_buffer
        misc = game_state.game_variables

        if more_perception:

            img0 = np.rollaxis(x_t1, 0, 3)
            #npimg = np.round(255 * img0)
            img = Image.fromarray(img0, 'RGB')

            # img.save("state.jpg")
            depth_t1 = predict_depth_map(img, sess, input_node, net)[0, :, :, 0]
            # depth_t1 = predict_segmentation(img)

            if False: #True: #False: #True:  # vizualization
                fig = plt.figure()
                fig.add_subplot(121)
                plt.imshow(img0, interpolation='nearest')
                fig.add_subplot(122)
                plt.imshow(depth_t1, interpolation='nearest')
                #fig.colorbar(ii)
                plt.show()
                fig.savefig("viz_depth"+"_"+str(t)+"_"+".jpg")

            x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))

            depth_t1 = transform.resize(depth_t1, (img_rows, img_cols))
            depth_t1 = (depth_t1 - np.min(depth_t1))/(np.max(depth_t1)-np.min(depth_t1))

            if False: #True: #False: #True:  # vizualization
                fig = plt.figure()
                fig.add_subplot(121)
                plt.imshow(x_t1, interpolation='nearest')
                fig.add_subplot(122)
                plt.imshow(depth_t1, interpolation='nearest')
                #fig.colorbar(ii)
                plt.show()
                fig.savefig("viz_depth"+"_"+str(t)+"_"+".jpg")

            # img = Image.open("demo_nyud_rgb.jpg")
            # img0 = np.array(img).astype('float32')
            # print(img0)
            # npimg = np.round(255 * img0)
            # img = Image.fromarray(img0, 'RGB')

            p_t1 = np.zeros((img_rows, img_cols, 2))
            p_t1[:,:,0] = x_t1
            p_t1[:,:,1] = depth_t1
            p_t1 = np.expand_dims(p_t1, axis=0) # 1x64x64x2

            s_t1 = p_t1

        else:
            x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
            x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
            s_t1 = x_t1


        if (prev_misc[0] - misc[0] > 8): # Pick up Poison
            poison += 1
        if (misc[0] > prev_misc[0]): # Pick up Health Pack
            medkit += 1

        previous_life = life
        if (is_terminated):
            life = 0
        else:
            life += 1

        # Update the cache
        prev_misc = misc

        if not test_phase:
            # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
            agent.replay_memory(s_t, action_idx, r_t, s_t1, m_t, is_terminated)

        if n_measures==3:
            m_t = np.array([misc[0] / 30.0, medkit/10.0, poison]) # Measurement after transition
        elif n_measures == 1:
            m_t = np.array([misc[0] / 30.0])

        if t > agent.observe and t % agent.timestep_per_train == 0 and not test_phase:
            # print("DO TRAIN")
            loss = agent.train_minibatch_replay(goal)

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0 and not test_phase:
            agent.save_model('../../experiments/'+title+'/model/DFP.h5')

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore/train" #train mais on continue à explorer
        else:
            state = "exploit/train" #train que en exploitant

        if test_phase:
            state = "test"

        if (is_terminated):
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ Medkit", medkit, "/ Poison", poison, "/ LIFE", max_life, "/ LOSS", loss)

            if GAME % agent.stats_window_size == 0 and t > agent.observe:
               mean_life = np.mean(np.array(life_buffer))
               var_life = np.var(np.array(life_buffer))
            else:
               mean_life = None
               var_life = None
            path_result = '../../experiments/' + title + '/logs/' + 'results.csv'

            if test_phase:
                path_result = '../../experiments/' + title + '/logs/' + 'results_test.csv'

            with open(path_result, mode='a') as log_file:
                writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([t, state, agent.epsilon, action_idx, r_t,
                                 medkit, poison, frags, amo, max_life, previous_life,
                                 mean_life, var_life, loss])

            medkit = 0
            poison = 0

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))

                # Reset rolling stats buffer
                life_buffer = []

                # Write Rolling Statistics to file
                with open('../../experiments/'+title+'/statistics/stats.txt', 'w+') as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')

        if t == tend:
            break
    sess.close()
    sess2.close()
