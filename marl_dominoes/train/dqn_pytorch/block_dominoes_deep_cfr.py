"""DQN agents trained on block dominoes by independent Q-learning using pytorch.

uses the `rl_environment.Environment` class to interact with the game.
"""

from absl import app
from absl import flags
from absl import logging
import numpy as np
import matplotlib.pyplot as plt
import sys

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import random_agent
import pyspiel

import open_spiel.python.games
import math
import os

from marl_dominoes.eval.eval import eval_against_random_bots
import torch

_MAX_WIDTH = int(os.getenv("COLUMNS", "80"))  # Get your TTY width.
FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "python_block_dominoes", "Name of the game")
flags.DEFINE_string("checkpoint_dir", "/Users/brunozorrilla/Documents/GitHub/marl_dominoes/marl_dominoes/agents/dqn_pytorch/q_network",
                    "Directory to save/load the agent.") 
flags.DEFINE_integer(
    "save_every", int(1e1),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e3),
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


def main(_):
  num_players = 2
  game = pyspiel.load_game(FLAGS.game_name)
  env = rl_environment.Environment(game=game)
  num_actions = game.num_distinct_actions()
  info_state_size = game.information_state_tensor_size()

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  agents = [
    dqn.DQN(
        player_id=idx,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        batch_size=FLAGS.batch_size) for idx in range(num_players)
  ]
  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      r_mean = eval_against_random_bots(env, agents, 1)
      logging.info("[%s] Mean episode rewards %s", ep + 1, r_mean)

    if (ep + 1) % FLAGS.save_every == 0:
        for i in range(len(agents)):
            agents[i].save(FLAGS.checkpoint_dir + f'{i}.pt')

    time_step = env.reset()
    while not time_step.last():
        player = time_step.observations["current_player"]
        agent_output = agents[player].step(time_step)
        time_step = env.step([agent_output.action])
        # print (f"Player: {player}, action: {agent_output.action}")

    # Episode is over, step all agents with final info state.
    for agent in agents:
        agent.step(time_step)

    # print("\n-=- Game over -=-\n")
    # print(f"Terminal state: {time_step}")
    # print(f"Returns: {time_step.rewards}")
  return


if __name__ == "__main__":
  app.run(main)
