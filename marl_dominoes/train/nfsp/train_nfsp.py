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

"""NFSP agents trained on Leduc Poker."""

from typing import Callable, Optional, Protocol, Union
import pandas as pd
import numpy as np

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import random_agent
import open_spiel.python.games 

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "python_block_dominoes", "Name of the game.")
flags.DEFINE_integer("num_players", 2,
                     "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(20e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 1000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 100_000,
                     "Episode frequency at which the agents are saved.")
flags.DEFINE_string("checkpoint_dir", "/tmp/", 
                    "Directory to save/load the agent models.")
flags.DEFINE_string("results_dir", "open_spiel/python/examples/tiny_block_dominoes/results/train/", 
                    "Directory to save the data.")
flags.DEFINE_bool("use_checkpoints", True, "load neural network weights.")


# NFSP model hyperparameters
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")



def main(unused_argv):
  logging.info("Loading %s", FLAGS.game)
  game_name = FLAGS.game
  num_players = FLAGS.num_players
  alg_name = "nfsp"
  df = pd.DataFrame({})
  rewards = []

  env = rl_environment.Environment(game_name)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "anticipatory_param": FLAGS.anticipatory_param,
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": FLAGS.epsilon_start,
      "epsilon_end": FLAGS.epsilon_end,
  }
  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    sess.run(tf.global_variables_initializer())

    if FLAGS.use_checkpoints:
      for agent in agents:
        if agent.has_checkpoint(FLAGS.checkpoint_dir):
          agent.restore(FLAGS.checkpoint_dir)
          logging.info("Loaded checkpoint from '%s'", FLAGS.checkpoint_dir)
        else:
          logging.info("No checkpoint found in '%s'", FLAGS.checkpoint_dir)

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        # losses = [agent.loss for agent in agents]
        r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
        # logging.info("Losses: %s", losses)
        rewards.append(r_mean)
        logging.info("[%s] Mean episode rewards %s", ep, r_mean)
        plot_rewards(rewards)
        df = pd.concat([df, pd.DataFrame(log_info(ep, r_mean))], ignore_index=True)
        df.to_csv(FLAGS.results_dir + f"{alg_name}_{game_name}_{FLAGS.num_train_episodes}_29-05.csv")
      if (ep + 1) % FLAGS.save_every == 0:
        for agent in agents:
          agent.save(FLAGS.checkpoint_dir + f"{alg_name}_{game_name}")
        logging.info(f"[{ep + 1}] Saved {alg_name} learner")

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


# plot it and save it
def log_info(ep, r_mean) -> dict[str, list[Union[float, str]]]:
  return {
    "Iteration": [ep+1],
    "Rewards": [r_mean]
  }
  

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
  return sum_episode_rewards / num_episodes

def plot_rewards(rewards):
            # plot the rewards
            fig = plt.figure()
            plt.plot([r[0] for r in rewards], color='blue', label='Starting agent')  # First reward in blue
            plt.plot([r[1] for r in rewards], color='red', label='Agent 2')  # Second reward in red
            plt.xlabel('Training Iterations')
            plt.ylabel('Reward')
            plt.title(f'{FLAGS.game}: Reward vs Training Iterations')
            plt.legend()
            plt.savefig(f'open_spiel/python/examples/saved_examples/{FLAGS.game}_reward_plot_deepcfr-08-05.png')
        

class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = list(range(FLAGS.num_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
        "info_state": [None] * FLAGS.num_players,
        "legal_actions": [None] * FLAGS.num_players
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


if __name__ == "__main__":
  app.run(main)
