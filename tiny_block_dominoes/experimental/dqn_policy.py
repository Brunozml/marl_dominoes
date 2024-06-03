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

"""DQN agents trained on tiny version of block dominoes, using jax implementation."""

from absl import app
from absl import flags
from absl import logging

import open_spiel.python.games
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import dqn
# from open_spiel.python.pytorch import dqn
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "python_block_dominoes", "Name of the game.")
flags.DEFINE_integer("num_train_episodes", int(3e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")


# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e4),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


class DQNPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, dqn_policies):
    game = env.game
    player_ids = [0, 1]
    super(DQNPolicies, self).__init__(game, player_ids)
    self._policies = dqn_policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    # with self._policies[cur_player].temp_mode_as(self._mode):
    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(unused_argv):
  game = FLAGS.game
  env = rl_environment.Environment(game)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  num_players = env.num_players

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  # agents = [
  #     dqn.DQN(
  #         player_id=idx,
  #         state_representation_size=info_state_size,
  #         num_actions=num_actions,
  #         hidden_layers_sizes=hidden_layers_sizes,
  #         replay_buffer_capacity=FLAGS.replay_buffer_capacity,
  #         batch_size=FLAGS.batch_size) for idx in range(num_players)
  # ]
  agents = [
      dqn.DQN(  # pylint: disable=g-complex-comprehension
          player_id,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          hidden_layers_sizes= hidden_layers_sizes,
          replay_buffer_capacity=FLAGS.replay_buffer_capacity,
          batch_size=FLAGS.batch_size
          ) 
          for player_id in [0, 1]
  ]



  expl_policies_avg = DQNPolicies(env, agents)

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      losses = [agent.loss for agent in agents]
      logging.info("Losses: %s", losses)
      expl = exploitability.exploitability(env.game, expl_policies_avg)
      logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
      logging.info("_____________________________________________")

    time_step = env.reset()
    while not time_step.last():
      current_player = time_step.observations["current_player"]
      current_agent = agents[current_player]
      agent_output = current_agent.step(time_step)
      time_step = env.step([agent_output.action])

    for agent in agents:
      agent.step(time_step)

if __name__ == "__main__":
  app.run(main)
