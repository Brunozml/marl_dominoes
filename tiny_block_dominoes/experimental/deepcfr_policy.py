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

"""Python Deep CFR example."""

import collections
from typing import Callable, Optional, Protocol, Union
import pandas as pd
import pickle

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
# from open_spiel.python.jax import deep_cfr
from open_spiel.python.pytorch import deep_cfr
import pyspiel

FLAGS = flags.FLAGS


flags.DEFINE_string("game_name", "python_block_dominoes", "Name of the game")
flags.DEFINE_integer("num_iterations", 200, "Number of training iterations.")
flags.DEFINE_integer("eval_every", 10,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 20,
                     "Episode frequency at which the agents are saved.")
flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/saved_examples/tiny_block_dominoes/agents/", 
                    "Directory to save/load the agent models.")
flags.DEFINE_string("results_dir", "open_spiel/python/examples/saved_examples/tiny_block_dominoes/results/", 
                    "Directory to save the data.")

# Deep CFR model hyper-parameters
# training parameters
flags.DEFINE_integer("num_traversals", 15, "Number of traversals/games") # used to be 40
flags.DEFINE_integer("batch_size_advantage", 2048, "Adv fn batch size")
flags.DEFINE_integer("batch_size_strategy", 2048, "Strategy batch size")
flags.DEFINE_integer("num_hidden", 64, "Hidden units in each layer")
flags.DEFINE_integer("num_layers", 3, "Depth of neural networks")
flags.DEFINE_bool("reinitialize_advantage_networks", False,
                  "Re-init value net on each CFR iter")
flags.DEFINE_float("learning_rate", 1e-3, "Optimizer learning rate")
flags.DEFINE_integer("memory_capacity",
                     1_000_000, "replay buffer capacity")
flags.DEFINE_integer("policy_network_train_steps",
                     5000, "training steps per iter")
flags.DEFINE_integer("advantage_network_train_steps",
                     750, "training steps per iter")


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)
  df = pd.DataFrame({})
  alg_name = "deepcfr"

  learner = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=tuple(
                [FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
      advantage_network_layers=tuple(
                [FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=FLAGS.learning_rate,
      batch_size_advantage=FLAGS.batch_size_advantage,
      batch_size_strategy=FLAGS.batch_size_strategy,
      memory_capacity= FLAGS.memory_capacity,
      policy_network_train_steps= FLAGS.policy_network_train_steps,
      advantage_network_train_steps=FLAGS.advantage_network_train_steps,
      reinitialize_advantage_networks=True
    )
  
  for ep in range(learner._num_iterations):
    solve(learner) #iterate over the training loop
    if (ep + 1) % FLAGS.eval_every == 0:
      average_policy = policy.tabular_policy_from_callable(game, learner.action_probabilities)
      expl = exploitability.exploitability(game, average_policy)
      df = pd.concat([df, pd.DataFrame(log_info(ep, expl))], ignore_index=True)
      df.to_csv(FLAGS.results_dir + f"deepcfr_{FLAGS.game_name}.csv")
                
    if (ep + 1) % FLAGS.save_every == 0:
      with open(FLAGS.checkpoint_dir + f"{alg_name}_{FLAGS.game_name}_{ep + 1}.pkl", "wb") as f:
        pickle.dump(learner, f)
        logging.info(f"[{ep + 1}] Saved {alg_name} learner")


# auxiliary functions
def solve(self):
    """Modified deep-cfr solution logic for online policy evaluation"""
    advantage_losses = collections.defaultdict(list)
    for _ in range(self._num_iterations):
        for p in range(self._num_players):
            for _ in range(self._num_traversals):
                self._traverse_game_tree(self._root_node, p)
            if self._reinitialize_advantage_networks:
                # Re-initialize advantage network for player and train from scratch.
                self.reinitialize_advantage_network(p)
            advantage_losses[p].append(self._learn_advantage_network(p))
        self._iteration += 1
    # Train policy network.
    policy_loss = self._learn_strategy_network()
    return self._policy_network, advantage_losses, policy_loss


# plot it and save it
def log_info(ep, expl) -> dict[str, list[Union[float, str]]]:
  logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
  return {
    "Iteration": [ep+1],
    "Exploitability": [expl]
  }


if __name__ == "__main__":
  app.run(main)
