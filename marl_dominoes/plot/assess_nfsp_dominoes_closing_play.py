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
from open_spiel.python.algorithms import nfsp
from open_spiel.python.games import block_dominoes
import pyspiel

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
  df = pd.DataFrame(columns = ['action', 'prob', 'state'])

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
  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    sess.run(tf.global_variables_initializer())

    if FLAGS.use_checkpoints:
      full_path = f"{FLAGS.checkpoint_dir}{alg_name}_{game_name}"
      logging.info("Looking for checkpoint in '%s'", full_path)
      for agent in agents:
        if agent.has_checkpoint(full_path):
          agent.restore(full_path)
          logging.info("Loaded checkpoint from '%s'", full_path)
        else:
          logging.info("No checkpoint found in '%s'", full_path)


    # look at initial action probabilities of the agents
    game = pyspiel.load_game(FLAGS.game)

    # Create the initial state
    state = game.new_initial_state()
    hand0 = [
        (0.0, 0.0),
        (0.0, 3.0),
        (0.0, 4.0),
        (0.0, 5.0),
        (0.0, 6.0),
        (5.0, 5.0),
        (5.0, 6.0),
    ]
    hand1 = [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (2.0, 6.0),
        (3.0, 5.0),
        (4.0, 5.0),
        (4.0, 6.0),
    ]
    deal_hands(state, [hand0, hand1])

    apply_action(state, block_dominoes.Action(0, (0.0, 0.0), None))
    # player 1 passes; has no tile with 0
    apply_action(state, block_dominoes.Action(0, (0.0, 6.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (4.0, 6.0), 6.0))
    apply_action(state, block_dominoes.Action(0, (0.0, 4.0), 4.0))
    # player 1 passes again (both edges 0)
    apply_action(state, block_dominoes.Action(0, (0.0, 3.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (3.0, 5.0), 3.0))

    print(state)
    for action, probs in joint_avg_policy.action_probabilities(state):
      print(f"Action: {state.action_to_string(state.current_player(), action)}, Prob: {probs}")

    # Output:
    # Action: p0 tile:(5.0, 6.0) pip:5.0, Prob: 0.540201683734727
    # Action: p0 tile:(5.0, 5.0) pip:5.0, Prob: 0.3557210499349776
    # Action: p0 tile:(0.0, 5.0) pip:0.0, Prob: 0.09545989634908883
    # Action: p0 tile:(0.0, 5.0) pip:5.0, Prob: 0.008617369981206574
        
    print("New Game!\n")
    # Create the initial state
    state = game.new_initial_state()
    hand0 = [
        (0.0, 0.0),
        (0.0, 3.0),
        (0.0, 4.0),
        (0.0, 5.0),
        (0.0, 6.0),
        (1.0, 5.0),
        (2.0, 2.0),
    ]
    hand1 = [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (2.0, 6.0),
        (3.0, 5.0),
        (4.0, 5.0),
        (4.0, 6.0),
    ]
    deal_hands(state, [hand0, hand1])

    apply_action(state, block_dominoes.Action(0, (0.0, 0.0), None))
    # player 1 passes; has no tile with 0
    apply_action(state, block_dominoes.Action(0, (0.0, 6.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (4.0, 6.0), 6.0))
    apply_action(state, block_dominoes.Action(0, (0.0, 4.0), 4.0))
    # player 1 passes again (both edges 0)
    apply_action(state, block_dominoes.Action(0, (0.0, 3.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (3.0, 5.0), 3.0))

    print(state)
    for action, probs in joint_avg_policy.action_probabilities(state):
      print(f"Action: {state.action_to_string(state.current_player(), action)}, Prob: {probs}")


    # Output: Closing is fine now!!
    # Action: p0 tile:(0.0, 5.0) pip:5.0, Prob: 0.5441169881582567
    # Action: p0 tile:(0.0, 5.0) pip:0.0, Prob: 0.3838916303392849
    # Action: p0 tile:(1.0, 5.0) pip:5.0, Prob: 0.07199138150245842

    print("New Game!\n")
    # Create the initial state
    state = game.new_initial_state()
    hand0 = [
        (0.0, 0.0),
        (0.0, 3.0),
        (0.0, 4.0),
        (0.0, 5.0),
        (0.0, 6.0),
        (1.0, 5.0),
        (2.0, 5.0),
    ]
    hand1 = [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (2.0, 6.0),
        (3.0, 5.0),
        (4.0, 5.0),
        (4.0, 6.0),
    ]
    deal_hands(state, [hand0, hand1])

    apply_action(state, block_dominoes.Action(0, (0.0, 0.0), None))
    # player 1 passes; has no tile with 0
    apply_action(state, block_dominoes.Action(0, (0.0, 6.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (4.0, 6.0), 6.0))
    apply_action(state, block_dominoes.Action(0, (0.0, 4.0), 4.0))
    # player 1 passes again (both edges 0)
    apply_action(state, block_dominoes.Action(0, (0.0, 3.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (3.0, 5.0), 3.0))

    print(state)
    for action, probs in joint_avg_policy.action_probabilities(state):
      print(f"Action: {state.action_to_string(state.current_player(), action)}, Prob: {probs}")

    # Output: Closing is fine now too!!
    # Action: p0 tile:(0.0, 5.0) pip:5.0, Prob: 0.42270917431123933
    # Action: p0 tile:(0.0, 5.0) pip:0.0, Prob: 0.3393534782299569
    # Action: p0 tile:(1.0, 5.0) pip:5.0, Prob: 0.16880019017970896
    # Action: p0 tile:(2.0, 5.0) pip:5.0, Prob: 0.06913715727909481

    print("New Game!\n")
    # Create the initial state
    state = game.new_initial_state()
    hand0 = [
        (0.0, 0.0),
        (0.0, 3.0),
        (0.0, 4.0),
        (0.0, 5.0),
        (0.0, 6.0),
        (2.0, 5.0),
        (5.0, 5.0),
    ]
    hand1 = [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (2.0, 6.0),
        (3.0, 5.0),
        (4.0, 5.0),
        (4.0, 6.0),
    ]
    deal_hands(state, [hand0, hand1])

    apply_action(state, block_dominoes.Action(0, (0.0, 0.0), None))
    # player 1 passes; has no tile with 0
    apply_action(state, block_dominoes.Action(0, (0.0, 6.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (4.0, 6.0), 6.0))
    apply_action(state, block_dominoes.Action(0, (0.0, 4.0), 4.0))
    # player 1 passes again (both edges 0)
    apply_action(state, block_dominoes.Action(0, (0.0, 3.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (3.0, 5.0), 3.0))

    print(state)
    for action, probs in joint_avg_policy.action_probabilities(state):
      print(f"Action: {state.action_to_string(state.current_player(), action)}, Prob: {probs}")
    
    # Output: I have a lot of 5s, lets use them
    # Action: p0 tile:(0.0, 5.0) pip:0.0, Prob: 0.5707628846434515 ## I lock both side with 5
    # Action: p0 tile:(0.0, 5.0) pip:5.0, Prob: 0.21599412863569672
    # Action: p0 tile:(2.0, 5.0) pip:5.0, Prob: 0.11719124760800195
    # Action: p0 tile:(5.0, 5.0) pip:5.0, Prob: 0.0960517391128498

    print("New Game!\n")
    # Create the initial state
    state = game.new_initial_state()
    hand0 = [
        (0.0, 0.0),
        (0.0, 3.0),
        (0.0, 4.0),
        (0.0, 5.0),
        (0.0, 6.0),
        (6.0, 6.0),
        (5.0, 5.0),
    ]
    hand1 = [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 4.0),
        (2.0, 6.0),
        (3.0, 5.0),
        (4.0, 5.0),
        (4.0, 6.0),
    ]
    deal_hands(state, [hand0, hand1])

    apply_action(state, block_dominoes.Action(0, (0.0, 0.0), None))
    # player 1 passes; has no tile with 0
    apply_action(state, block_dominoes.Action(0, (0.0, 6.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (4.0, 6.0), 6.0))
    apply_action(state, block_dominoes.Action(0, (0.0, 4.0), 4.0))
    # player 1 passes again (both edges 0)
    apply_action(state, block_dominoes.Action(0, (0.0, 3.0), 0.0))
    apply_action(state, block_dominoes.Action(1, (3.0, 5.0), 3.0))

    print(state)
    for action, probs in joint_avg_policy.action_probabilities(state):
      print(f"Action: {state.action_to_string(state.current_player(), action)}, Prob: {probs}")
    



def deal_hands(state, hands):
  deck = block_dominoes._DECK
  for hand in hands:
    for t in hand:
      state.apply_action(deck.index(t))

def apply_action(state, action):
  actions_str = block_dominoes._ACTIONS_STR
  state.apply_action(actions_str.index(str(action)))


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
        
    # Sort the dictionary by values in descending order and convert it to a list of tuples
    sorted_prob_list = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_prob_list
  
  def all_action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
      state.information_state_tensor(cur_player))

    info_state = rl_environment.TimeStep(
      observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs

    return p


if __name__ == "__main__":
  app.run(main)
