# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""Run methods for training a DQN agent on Atari.

Methods in this module are usually referenced by |train.py|.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import time
import copy
from third_party.dopamine import checkpointer
from third_party.dopamine import iteration_statistics
import dqn_agent
import ToM_agent
import gin.tf
#from hanabi_learning_environment import rl_env
import numpy as np
import rainbow_agent
import tensorflow as tf
import sys
sys.path.append('/code/CoRE_Project/hanabi_learning_environment')
import rl_env
from ToMmodel import dataset

LENIENT_SCORE = False


class ObservationStacker(object):
  """Class for stacking agent observations."""

  def __init__(self, history_size, observation_size, num_players):
    """Initializer for observation stacker.

    Args:
      history_size: int, number of time steps to stack.
      observation_size: int, size of observation vector on one time step.
      num_players: int, number of players.
    """
    
    ## 注意，希望在my_agent.py里面继承ToM_dqn_agent，增加ToM模块的ToM_infer_states， 不在训练代码中提示。
    ## 具体见151行
    self._history_size = history_size
    self._observation_size = observation_size
    self._num_players = num_players
    self._obs_stacks = list()
    for _ in range(0, self._num_players):
      self._obs_stacks.append(np.zeros(self._observation_size *
                                       self._history_size))

  def add_observation(self, observation, current_player):
    """Adds observation for the current player.

    Args:
      observation: observation vector for current player.
      current_player: int, current player id.
    """
    self._obs_stacks[current_player] = np.roll(self._obs_stacks[current_player],
                                               -self._observation_size)
    self._obs_stacks[current_player][(self._history_size - 1) *
                                     self._observation_size:] = observation

  def get_observation_stack(self, current_player):
    """Returns the stacked observation for current player.

    Args:
      current_player: int, current player id.
    """

    return self._obs_stacks[current_player]

  def reset_stack(self):
    """Resets the observation stacks to all zero."""

    for i in range(0, self._num_players):
      self._obs_stacks[i].fill(0.0)

  @property
  def history_size(self):
    """Returns number of steps to stack."""
    return self._history_size

  def observation_size(self):
    """Returns the size of the observation vector after history stacking."""
    return self._observation_size * self._history_size


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: A list of paths to the gin configuration files for this
      experiment.
    gin_bindings: List of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_environment(game_type='Hanabi-Full', num_players=2):
  """Creates the Hanabi environment.

  Args:
    game_type: Type of game to play. Currently the following are supported:
      Hanabi-Full: Regular game.
      Hanabi-Small: The small version of Hanabi, with 2 cards and 2 colours.
    num_players: Int, number of players to play this game.

  Returns:
    A Hanabi environment.
  """
  return rl_env.make(
      environment_name=game_type, num_players=num_players, pyhanabi_path=None)


@gin.configurable
def create_obs_stacker(environment, history_size=4, hcic=False):
  """Creates an observation stacker.

  Args:
    environment: environment object.
    history_size: int, number of steps to stack.

  Returns:
    An observation stacker object.
  """

  return ObservationStacker(history_size,
                            environment.vectorized_observation_shape()[0],
                            environment.players)


@gin.configurable
def create_agent(environment, obs_stacker, agent_type='Rainbow', hcic=False, goir=False, path_to_pb=None):
  """Creates the Hanabi agent.

  Args:
    environment: The environment.
    obs_stacker: Observation stacker object.
    agent_type: str, type of agent to construct.

  Returns:
    An agent for playing Hanabi.

  Raises:
    ValueError: if an unknown agent type is requested.
  """
  if agent_type == 'DQN':
    return dqn_agent.DQNAgent(observation_size=obs_stacker.observation_size(),
                              num_actions=environment.num_moves(),
                              num_players=environment.players)
  elif agent_type == 'Rainbow':
    observation_size = obs_stacker.observation_size()
    if hcic:
      observation_size = observation_size + 5 * 25 # environment.game.hand_size() * 25

    return rainbow_agent.RainbowAgent(
      observation_size=observation_size,
      num_actions=environment.num_moves(),
      num_players=environment.players,
      hcic=hcic,
      goir=goir,
      path_to_pb=path_to_pb)
    
  elif agent_type == 'ToM_Agent':
    return ToM_agent.ToMAgent(
      observation_size=obs_stacker.observation_size(),
        num_actions=environment.num_moves(),
        num_players=environment.players)
    
  else:
    raise ValueError('Expected valid agent_type, got {}'.format(agent_type))


def initialize_checkpointing(agent, experiment_logger, checkpoint_dir,
                             checkpoint_file_prefix='ckpt'):
  """Reloads the latest checkpoint if it exists.

  The following steps will be taken:
   - This method will first create a Checkpointer object, which will be used in
     the method and then returned to the caller for later use.
   - It will then call checkpointer.get_latest_checkpoint_number to determine
     whether there is a valid checkpoint in checkpoint_dir, and what is the
     largest file number.
   - If a valid checkpoint file is found, it will load the bundled data from
     this file and will pass it to the agent for it to reload its data.
   - If the agent is able to successfully unbundle, this method will verify that
     the unbundled data contains the keys, 'logs' and 'current_iteration'. It
     will then load the Logger's data from the bundle, and will return the
     iteration number keyed by 'current_iteration' as one of the return values
     (along with the Checkpointer object).

  Args:
    agent: The agent that will unbundle the checkpoint from checkpoint_dir.
    experiment_logger: The Logger object that will be loaded from the
      checkpoint.
    checkpoint_dir: str, the directory containing the checkpoints.
    checkpoint_file_prefix: str, the checkpoint file prefix.

  Returns:
    start_iteration: int, The iteration number to start the experiment from.
    experiment_checkpointer: The experiment checkpointer.
  """
  experiment_checkpointer = checkpointer.Checkpointer(
      checkpoint_dir, checkpoint_file_prefix)

  start_iteration = 0

  # Check if checkpoint exists. Note that the existence of checkpoint 0 means
  # that we have finished iteration 0 (so we will start from iteration 1).
  latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
      checkpoint_dir)
  if latest_checkpoint_version >= 0:
    dqn_dictionary = experiment_checkpointer.load_checkpoint(
        latest_checkpoint_version)
    if agent.unbundle(
        checkpoint_dir, latest_checkpoint_version, dqn_dictionary):
      assert 'logs' in dqn_dictionary
      assert 'current_iteration' in dqn_dictionary
      experiment_logger.data = dqn_dictionary['logs']
      start_iteration = dqn_dictionary['current_iteration'] + 1
      try:
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',start_iteration)
      except:
        tf.compat.v1.logging.info('Reloaded checkpoint and will start from iteration %d',start_iteration)

  return start_iteration, experiment_checkpointer


def format_legal_moves(legal_moves, action_dim):
  """Returns formatted legal moves.

  This function takes a list of actions and converts it into a fixed size vector
  of size action_dim. If an action is legal, its position is set to 0 and -Inf
  otherwise.
  Ex: legal_moves = [0, 1, 3], action_dim = 5
      returns [0, 0, -Inf, 0, -Inf]

  Args:
    legal_moves: list of legal actions.
    action_dim: int, number of actions.

  Returns:
    a vector of size action_dim.
  """
  new_legal_moves = np.full(action_dim, -float('inf'))
  if legal_moves:
    new_legal_moves[legal_moves] = 0
  return new_legal_moves


def parse_observations(observations, num_actions, obs_stacker):
  """Deconstructs the rich observation data into relevant components.

  Args:
    observations: dict, containing full observations.
    num_actions: int, The number of available actions.
    obs_stacker: Observation stacker object.

  Returns:
    current_player: int, Whose turn it is.
    legal_moves: `np.array` of floats, of length num_actions, whose elements
      are -inf for indices corresponding to illegal moves and 0, for those
      corresponding to legal moves.
    observation_vector: Vectorized observation for the current player.
  """
  current_player = observations['current_player']
  current_player_observation = (
      observations['player_observations'][current_player])

  legal_moves = current_player_observation['legal_moves_as_int']
  legal_moves = format_legal_moves(legal_moves, num_actions)

  observation_vector = current_player_observation['vectorized']
  obs_stacker.add_observation(observation_vector, current_player)
  observation_vector = obs_stacker.get_observation_stack(current_player)

  return current_player, legal_moves, observation_vector
def parse_state_action(environment, observations, action):
  """ 为了保存成json文件，对observations和根据observation产生的action进行合并，返回一个dict """
  '''
      如果eval的话,需要记录observations和 actions。 先有动作再有observation,记住。
      若action是int则需要转换(RL 是 int 的):
        在这里  observation is a dict
        action是一个int, 需要转换成 HanabiMove type。这个class在pyhanabi.py中有wrap,原始的定义见
        hanabi_lib/hanabi_move.h, hanabi_state.h, hanabi_game.h 以及对应.cc文件
        
        如需要引用c定义的lib, 使用ffi可以在py中使用c的库函数,样例见pyhanabi.py
        ## notice! move 在pyhanabi.py有wrapper,因此可以用pyhanabi.py的函数来做!!!!!  line 296
        动作type定义见pyhanabi.py line 286 && hanabi_lib/hanabi_move.h line 21 底下注释 
  '''
  ## 获取HanabiMove类型动作定义，为转dict做准备
  move = environment.game.get_move(action.item())  ##理论上是HanabiMove type的class 参见hanabi_move.h.
  ## 调用包装的函数，获取类型，具体定义见 pyhanabi.py line 286 && hanabi_lib/hanabi_move.h line 21 底下注释 
  type_ = move.type()
  action_dictionary = {'action_type': None, 'card_index':None, 'color': None, 'rank': None, 'target_offset': None}  # hanabi_move.h, line 36
  ## 根据type生成action dictionary
  action_dictionary['action_type'] = int(type_)  ##都得加
  #if type_ ==0:         ### INVALID = 0
  if type_ == 1 or type_ == 2:                                  ### PLAY = 1; DISCARD = 2
    action_dictionary['card_index'] = move.card_index()
      
  elif type_ == 3:                                              ### REVEAL_COLOR = 3
    action_dictionary['color'] = move.color()
    action_dictionary['target_offset'] = move.card_index()
  elif type_ == 4:                                              ### REVEAL_RANK = 4
    action_dictionary['rank'] = move.rank()
    action_dictionary['target_offset'] = move.card_index()
  elif type_ == 5:                                              ### DEAL = 5
    action_dictionary['color'] = move.color()
    action_dictionary['rank'] = move.rank()
    action_dictionary['target_offset'] = move.card_index()
  else:                                                         ### 错误
    raise ValueError
      
  ##现在获得action_dictionary和 observations. 在observations中加入'player_action'这个信息：
  new_observe = {}
  '''
  问题：deepcopy无法作用多线程上
  observations['player_observations'][i]这个dict中存在'pyhanabi'是HanabiObservation类型的，不可存到json file中
  '''
  cur_player_obs_list = []
  for agent_dict in observations['player_observations']:  #n个dict
    keys = list(agent_dict.keys())
    cur_dict = {}
    for i in range(len(keys)):
      if keys[i] =='pyhanabi':  #HanabiObserve type
        continue
      cur_dict[keys[i]] = agent_dict[keys[i]]
    cur_player_obs_list.append(cur_dict)
  new_observe['player_observations'] = cur_player_obs_list
  new_observe['current_player'] = observations['current_player']
  new_observe['player_action'] = action_dictionary
  return new_observe
      

def run_one_episode(agent, environment, obs_stacker):
  """Runs the agent on a single game of Hanabi in self-play mode.

  Args:
    agent: Agent playing Hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.

  Returns:
    step_number: int, number of actions in this episode.
    total_reward: float, undiscounted return for this episode.
  """
  json_observe_list = []  ##如果eval,记录json数据
  obs_stacker.reset_stack()
  observations = environment.reset()   ##initial obs
  current_player, legal_moves, observation_vector = (
      parse_observations(observations, environment.num_moves(), obs_stacker))
  action = agent.begin_episode(current_player, legal_moves, observation_vector)  ##first action (658,)

  is_done = False
  total_reward = 0
  step_number = 0
  lookback = 10

  has_played = {current_player}

  # Keep track of per-player reward.
  reward_since_last_action = np.zeros(environment.players)
  
  ## 改了
  if agent.eval_mode == True or agent.hcic == True:
    json_observe_list.append(parse_state_action(environment, observations, action))
  
  while not is_done:
    observations, reward, is_done, _ = environment.step(action.item())  #.item目的是去除梯度（是int形式）
    
    modified_reward = max(reward, 0) if LENIENT_SCORE else reward
    total_reward += modified_reward

    reward_since_last_action += modified_reward

    step_number += 1
    if is_done:
      break
    current_player, legal_moves, observation_vector = (
        parse_observations(observations, environment.num_moves(), obs_stacker))
    if current_player in has_played:
      if agent.hcic == False or step_number < lookback:
        action = agent.step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector, observations)
      # yhx 加了传入observations(dict)，只在我出牌或弃牌时确认是什么牌，不用于获取仍在我手中的牌的信息
      else:
        act_seq = [{'current_player': x['current_player'], 'player_action': x['player_action']} for x in json_observe_list[-lookback:]]
        data = dataset.data_process(copy.deepcopy(act_seq), observations['player_observations'][current_player], current_player)
        action = agent.step(reward_since_last_action[current_player],
                          current_player, legal_moves, observation_vector, observations, data)
      
    else:
      # Each player begins the episode on their first turn (which may not be
      # the first move of the game).
      action = agent.begin_episode(current_player, legal_moves,
                                   observation_vector)
      has_played.add(current_player)

    # Reset this player's reward accumulator.
    reward_since_last_action[current_player] = 0
      
    if agent.eval_mode == True or agent.hcic == True:  ##先有 observations, 再有action
      json_observe_list.append(parse_state_action(environment, observations, action))

  agent.end_episode(reward_since_last_action)
  try:
    tf.logging.info('EPISODE: %d %g', step_number, total_reward)
  except:
    tf.compat.v1.logging.info('EPISODE: %d %g', step_number, total_reward)
    
  return step_number, total_reward, json_observe_list


def run_one_phase(agent, environment, obs_stacker, min_steps, statistics,
                  run_mode_str):
  """Runs the agent/environment loop until a desired number of steps.

  Args:
    agent: Agent playing hanabi.
    environment: environment object.
    obs_stacker: Observation stacker object.
    min_steps: int, minimum number of steps to generate in this phase.
    statistics: `IterationStatistics` object which records the experimental
      results.
    run_mode_str: str, describes the run mode for this agent.

  Returns:
    The number of steps taken in this phase, the sum of returns, and the
      number of episodes performed.
  """
  step_count = 0
  num_episodes = 0
  sum_returns = 0.

  while step_count < min_steps:  ## run_one episode, episode代表一轮游戏。一个step是一轮游戏。 phase是一场游戏
    episode_length, episode_return, _ = run_one_episode(agent, environment,
                                                     obs_stacker)
    statistics.append({
        '{}_episode_lengths'.format(run_mode_str): episode_length,
        '{}_episode_returns'.format(run_mode_str): episode_return
    })

    step_count += episode_length
    sum_returns += episode_return
    num_episodes += 1

  return step_count, sum_returns, num_episodes


@gin.configurable
def run_one_iteration(agent, environment, obs_stacker,
                      iteration, training_steps,
                      save_json_dir = None,  ##加了
                      evaluate_every_n=100,
                      num_evaluation_games=100):
  """Runs one iteration of agent/environment interaction.

  An iteration involves running several episodes until a certain number ofDQN'
  steps are obtained.

  Args:
    agent: Agent playing hanabi.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.
    iteration: int, current iteration number, used as a global_step.
    training_steps: int, the number of training steps to perform.
    evaluate_every_n: int, frequency of evaluation.
    num_evaluation_games: int, number of games per evaluation.

  Returns:
    A dict containing summary statistics for this iteration.
  """
  start_time = time.time()

  statistics = iteration_statistics.IterationStatistics()

  # First perform the training phase, during which the agent learns.
  agent.eval_mode = False
  number_steps, sum_returns, num_episodes = (
      run_one_phase(agent, environment, obs_stacker, training_steps, statistics,
                    'train'))
  time_delta = time.time() - start_time

  if tf.__version__[0] =='1':
    logger = tf.logging
  else:
    logger = tf.compat.v1.logging
    
  logger.info('Average training steps per second: %.2f',
                  number_steps / time_delta)

  average_return = sum_returns / num_episodes
  logger.info('Average per episode return: %.2f', average_return)
  statistics.append({'average_return': average_return})

  # Also run an evaluation phase if desired.
  if evaluate_every_n is not None and iteration % evaluate_every_n == 0:
    episode_data = []
    agent.eval_mode = True
    # Collect episode data for all games.
    for iter in range(num_evaluation_games):  ## run_one_episode是一局，一个episode是一局。 num_evaluation_steps是多少个测试游戏
      ## 改了，因为 run_one_episode还有存储json_dict
      step_number, total_reward, json_observe_list = run_one_episode(agent, environment, obs_stacker)
      #print(json_observe_list[0])
      episode_data.append(tuple((step_number, total_reward)))

      ## 根据 iter (测试游戏的局数)和 iteration(训练局数)来保存
      data_name = str(iteration)+'_eval_'+str(iter)+'.json'
      file_path = save_json_dir+data_name 
      ## 存
      with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_observe_list, indent=2))
        
    logger.info(f"successfully_save json files in iteration {iteration}.")
      
    eval_episode_length, eval_episode_return = map(np.mean, zip(*episode_data))

    statistics.append({
        'eval_episode_lengths': eval_episode_length,
        'eval_episode_returns': eval_episode_return
    })
    logger.info('Average eval. episode length: %.2f  Return: %.2f',
                    eval_episode_length, eval_episode_return)
  else:
    statistics.append({
        'eval_episode_lengths': -1,
        'eval_episode_returns': -1
    })

  return statistics.data_lists


def log_experiment(experiment_logger, iteration, statistics,
                   logging_file_prefix='log', log_every_n=1):
  """Records the results of the current iteration.

  Args:
    experiment_logger: A `Logger` object.
    iteration: int, iteration number.
    statistics: Object containing statistics to log.
    logging_file_prefix: str, prefix to use for the log files.
    log_every_n: int, specifies logging frequency.
  """
  if iteration % log_every_n == 0:
    experiment_logger['iter{:d}'.format(iteration)] = statistics
    experiment_logger.log_to_file(logging_file_prefix, iteration)


def checkpoint_experiment(experiment_checkpointer, agent, experiment_logger,
                          iteration, checkpoint_dir, checkpoint_every_n):
  """Checkpoint experiment data.

  Args:
    experiment_checkpointer: A `Checkpointer` object.
    agent: An RL agent.
    experiment_logger: a Logger object, to include its data in the checkpoint.
    iteration: int, iteration number for checkpointing.
    checkpoint_dir: str, the directory where to save checkpoints.
    checkpoint_every_n: int, the frequency for writing checkpoints.
  """
  if iteration % checkpoint_every_n == 0:
    agent_dictionary = agent.bundle_and_checkpoint(checkpoint_dir, iteration)
    if agent_dictionary:
      agent_dictionary['current_iteration'] = iteration
      agent_dictionary['logs'] = experiment_logger.data
      experiment_checkpointer.save_checkpoint(iteration, agent_dictionary)


@gin.configurable
def run_experiment(agent,
                   environment,
                   start_iteration,
                   obs_stacker,
                   experiment_logger,
                   experiment_checkpointer,
                   checkpoint_dir,
                   save_json_dir,  ##加了
                   num_iterations=200,
                   training_steps=5000,
                   logging_file_prefix='log',
                   log_every_n=1,
                   checkpoint_every_n=1):
  """Runs a full experiment, spread over multiple iterations."""
  if tf.__version__[0] == '1':
    logger = tf.logging
  else:
    logger = tf.compat.v1.logging
  logger.info('Beginning training...')
  if num_iterations <= start_iteration:
    logger.warning('num_iterations (%d) < start_iteration(%d)',
                       num_iterations, start_iteration)
    return

  for iteration in range(start_iteration, num_iterations):
    start_time = time.time()
    statistics = run_one_iteration(agent, environment, obs_stacker, iteration,
                                   training_steps, save_json_dir)
    logger.info('Iteration %d took %d seconds', iteration,
                    time.time() - start_time)
    start_time = time.time()
    log_experiment(experiment_logger, iteration, statistics,
                   logging_file_prefix, log_every_n)
    logger.info('Logging iteration %d took %d seconds', iteration,
                    time.time() - start_time)
    start_time = time.time()
    checkpoint_experiment(experiment_checkpointer, agent, experiment_logger,
                          iteration, checkpoint_dir, checkpoint_every_n)
    logger.info('Checkpointing iteration %d took %d seconds', iteration,
                    time.time() - start_time)
