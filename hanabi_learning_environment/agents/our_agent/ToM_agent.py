"""
    Implementation of 
    a ToM-based DQN agent WITH INTRINSIC LOSS 
    adapted to the multiplayer setting.
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from ToM_DQN import *

import gin.tf
import numpy as np
import replay_memory
import tensorflow as tf
import torch

@gin.configurable
class ToMAgent(DQNAgent):
    
    @gin.configurable
    def __init__(self,
               num_actions=None,
               observation_size=None,
               num_players=None,
               ):
        ## agent的eval_mode直接继承于基类
        ToM_infer_states_size =  NotImplementedError
        super.__init__(num_actions=num_actions, observation_size = observation_size, ToM_infer_states_size = ToM_infer_states_size, num_players = num_players)
        
    
    def begin_episode(self, current_player, legal_actions, observation):  ##不确定
        
        ToM_infer_states ## 从ToM model中来
        
        ## 基类包括 self._train_step() 相当于是forward() 函数， 返回一个action同时记录转移信息。
        # 因此在前面只需用ToM model生成ToM_infer_states并update这个model。两块tensor不连通，依照目前的设计
        
        return super().begin_episode(current_player, legal_actions, observation, ToM_infer_states)
    
    def step(self, reward, current_player, legal_actions, observation):
        
        ToM_infer_states ## 从ToM model中来
        
        ## 如上所述，self.step()和self.begin_episode()在ToM_DQN中没区别
        
        return super().step(reward, current_player, legal_actions, observation, ToM_infer_states)
    
    def end_episode(self, final_rewards):  ##不确定
        return super().end_episode(final_rewards)
    
    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):  ##不确定
        return super().unbundle(checkpoint_dir, iteration_number, bundle_dictionary)
    
    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number): ##不确定
        return super().bundle_and_checkpoint(checkpoint_dir, iteration_number)
    
    
    