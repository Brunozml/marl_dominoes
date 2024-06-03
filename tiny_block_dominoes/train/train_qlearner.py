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

"""Qlearning agents trained on tiny block dominoes."""
import pandas as pd
from typing import Union
import pickle

from absl import app
from absl import flags
from absl import logging


from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "python_block_dominoes", "Name of the game.")
flags.DEFINE_integer("num_train_episodes", int(3e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 100_000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", int(5e5),
                     "Episode frequency at which the agents are saved.")
flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/tiny_block_dominoes/agents/", 
                    "Directory to save/load the agent models.")
flags.DEFINE_string("results_dir", "open_spiel/python/examples/tiny_block_dominoes/results/train/", 
                    "Directory to save the data.")


class QLearnerPolicies(policy.TabularPolicy):
  """Joint policy to be evaluated."""

  def __init__(self, env, dqn_policies):
    game = env.game
    player_ids = [0, 1]
    super(QLearnerPolicies, self).__init__(game, player_ids)
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
  game_name = FLAGS.game
  num_players = 2
  df = pd.DataFrame({})
  env = rl_environment.Environment(game_name)
  num_actions = env.action_spec()["num_actions"]
  alg_name = "qlearner"

  agents = [
        tabular_qlearner.QLearner(
                        player_id=idx,
                        num_actions=num_actions,
                        centralized= False,
        )
        for idx in range(num_players)
  ]

  expl_policies_avg = QLearnerPolicies(env, agents)

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
       expl = exploitability.exploitability(env.game, expl_policies_avg)
        # save exploitability in csv
       df = pd.concat([df, pd.DataFrame(log_info(ep, expl))], ignore_index=True)
       df.to_csv(FLAGS.results_dir + f"{alg_name}_{game_name}.csv")
    if (ep + 1) % FLAGS.save_every == 0:
                with open(FLAGS.checkpoint_dir + f"{alg_name}_{game_name}_{ep + 1}.pkl", "wb") as f:
                  pickle.dump(expl_policies_avg, f)
                  logging.info(f"[{ep + 1}] Saved {alg_name} policy")

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

# plot it and save it
def log_info(ep, expl) -> dict[str, list[Union[float, str]]]:
  logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
  return {
    "Iteration": [ep+1],
    "Exploitability": [expl]
  }

if __name__ == "__main__":
  app.run(main)
