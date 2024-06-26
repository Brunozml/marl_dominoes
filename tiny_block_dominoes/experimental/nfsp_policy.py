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

"""NFSP agents trained on Leduc Poker. BUGGY. losses are not being computed for now."""

from typing import Callable, Optional, Protocol, Union
import pandas as pd
import pickle

from absl import app
from absl import flags
from absl import logging
# import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import nfsp

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "leduc_poker",
                    "Name of the game.")
flags.DEFINE_integer("num_players", 2,
                      "Number of players.")
flags.DEFINE_integer("num_train_episodes", 400,
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 200,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 1000,
                     "Episode frequency at which the agents are saved.")
flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/saved_examples/tiny_block_dominoes/agents/", 
                    "Directory to save/load the agent models.")
flags.DEFINE_string("results_dir", "open_spiel/python/examples/saved_examples/tiny_block_dominoes/results/", 
                    "Directory to save the data.")

flags.DEFINE_string("evaluation_metric", "nash_conv",
                    "Choose from 'exploitability', 'nash_conv'.")


# NFSP model hyper-parameters
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


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game)
  game_name = FLAGS.game
  df = pd.DataFrame({})
  # num_players = FLAGS.num_players

  # env_configs = {"players": num_players}
  env = rl_environment.Environment(game_name) # , **env_configs)
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

  agents = [
      nfsp.NFSP(idx, info_state_size, num_actions, hidden_layers_sizes,
                **kwargs) for idx in [0, 1]
  ]
  joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

  # if FLAGS.use_checkpoints:
  #   for agent in agents:
  #     if agent.has_checkpoint(FLAGS.checkpoint_dir):
  #       agent.restore(FLAGS.checkpoint_dir)

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      losses = [agent.loss for agent in agents]
      logging.info("Losses: %s", losses)
      expl = exploitability.exploitability(env.game, joint_avg_policy)
      df = pd.concat([df, pd.DataFrame(log_info(ep, expl))], ignore_index=True)
      df.to_csv(FLAGS.results_dir + f"nfsp_{game_name}.csv")

    if (ep + 1) % FLAGS.save_every == 0:
        with open(FLAGS.checkpoint_dir + f"nfsp_{game_name}_{ep + 1}.pkl", "wb") as f:
          pickle.dump(joint_avg_policy, f)
          logging.info(f"[{ep + 1}] Saved nfsp learner")

    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      action_list = [agent_output.action]
      time_step = env.step(action_list)

    # Episode is over, step all agents with final info state.
    for agent in agents:
      agent.step(time_step)


# auxiliary functions
def log_info(ep, expl) -> dict[str, list[Union[float, str]]]:
  logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
  return {
    "Iteration": [ep+1],
    "Exploitability": [expl]
  }
  

if __name__ == "__main__":
  app.run(main)
