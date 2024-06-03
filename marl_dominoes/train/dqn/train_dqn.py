# Copyright 2019 DeepMind Technologies Limited
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

"""DQN agents trained on block dominoes by independent Q-learning."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import sys

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent

import open_spiel.python.games
import math
import os

_MAX_WIDTH = int(os.getenv("COLUMNS", "80"))  # Get your TTY width.
FLAGS = flags.FLAGS

rewards = []
# "open_spiel/python/examples/saved_examples"
# Training parameters
flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/agents/dqn",
                    "Directory to save/load the agent.") # "/tmp/nfsp_test",
flags.DEFINE_integer(
    "save_every", int(1e2),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e4),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 100,
    "Episode frequency at which the DQN agents are evaluated.")
flags.DEFINE_boolean("interactive", False, "Whether to allow interactive play after training.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")

def pretty_board(env):
  """Returns the board in `time_step` in a human readable format."""
  # info_state = time_step.observations["info_state"][0]
  board = str(env.get_state)
  return board

def _print_columns(strings):
  """Returns a string of formatted columns."""
  padding = 2
  longest = max(len(s) for s in strings)
  max_columns = math.floor((_MAX_WIDTH - 1) / (longest + 2 * padding))
  rows = math.ceil(len(strings) / max_columns)
  columns = math.ceil(len(strings) / rows)  # Might not fill all max_columns.
  result = ""
  for r in range(rows):
    for c in range(columns):
      i = r + c * rows
      if i < len(strings):
        result += " " * padding + strings[i].ljust(longest + padding)
    result += "\n"
  return result

def pretty_actions(env):
  """Returns the legal actions in `time_step` in a human readable format."""
  state = env.get_state
  legal_actions = state.legal_actions(state.current_player())

  action_map = {
    state.action_to_string(state.current_player(), action): action
    for action in legal_actions
  }

  actions_string = ""
  longest_num = max(len(str(action)) for action in legal_actions)
  actions_string += _print_columns([
      "{}: {}".format(str(action).rjust(longest_num), action_str)
      for action_str, action in sorted(action_map.items())
    ])
  return actions_string

def plot_rewards(rewards):
  # plot the rewards
  fig = plt.figure()
  plt.plot([r[0] for r in rewards], color='blue', label='Agent 1')  # First reward in blue
  plt.plot([r[1] for r in rewards], color='red', label='Agent 2')  # Second reward in red
  plt.xlabel('Training Iterations')
  plt.ylabel('Reward')
  plt.title('Reward vs Training Iterations')
  plt.legend()
  plt.savefig('open_spiel/python/examples/saved_examples/reward_plot.png')
  plt.close(fig)  # Close the figure

def command_line_action(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1
  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue
  return action

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  
  rewards.append(sum_episode_rewards / num_episodes)
  return sum_episode_rewards / num_episodes


def main(_):
  game = "python_block_dominoes"
  num_players = 2

  # env_configs = {"columns": 5, "rows": 5}
  env = rl_environment.Environment(game)  # , **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  with tf.Session() as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    agents = [
        dqn.DQN(
            session=sess,
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size) for idx in range(num_players)
    ]

    sess.run(tf.global_variables_initializer())

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
        logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)
        plot_rewards(rewards)
      if (ep + 1) % FLAGS.save_every == 0:
        for agent in agents:
          agent.save(FLAGS.checkpoint_dir)

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = agents[player_id].step(time_step)
          action_list = [agent_output.action]
        else:
          agents_output = [agent.step(time_step) for agent in agents]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)
      
    if not FLAGS.interactive:
      return

        # 2. Play from the command line against the trained agent.
    human_player = 1
    while True:
      logging.info("You are playing as %s", "O" if human_player else "X")
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id == human_player:
          agent_out = agents[human_player].step(time_step, is_evaluation=True)
          logging.info("\n%s", agent_out.probs)
          logging.info("Game state:\n%s", pretty_board(env))
          logging.info("Legal actions: %s", pretty_actions(env) )
          action = command_line_action(time_step)
        else:
          agent_out = agents[1 - human_player].step(time_step, is_evaluation=True)
          action = agent_out.action
        time_step = env.step([action])

      logging.info("Final game state:\n%s", pretty_board(env))

      logging.info("End of game!")
      if time_step.rewards[human_player] > 0:
        logging.info("You win")
      elif time_step.rewards[human_player] < 0:
        logging.info("You lose")
      else:
        logging.info("Draw")
      # # Switch order of players
      # human_player = 1 - human_player
      break


if __name__ == "__main__":
  app.run(main)
