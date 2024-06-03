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
from open_spiel.python.algorithms import mmd_dilated
import pyspiel
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "python_block_dominoes", "Name of the game.")
flags.DEFINE_integer("num_train_episodes", 1000,
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 5,
                     "Episode frequency at which the agents are saved.")
flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/saved_examples/tiny_block_dominoes/agents/", 
                    "Directory to save/load the agent models.")


# MMD model hyper-parameters
hyperparameters = {
    "python_block_dominoes": {
        "annealing_temperature": {
            "temp_schedule": lambda i: 5 / np.sqrt(i),
            "lr_schedule": lambda i: 1 / np.sqrt(i), # alpha
            "mag_lr_schedule": lambda i: 0,
        },
        "moving_magnet": {
            "temp_schedule": lambda i: 1,
            "lr_schedule": lambda i: 0.1,
            "mag_lr_schedule": lambda i: 0.05,
        },
    },
}


def main(unused_argv):
  game_name = FLAGS.game
  game = pyspiel.load_game(game_name)
  env = rl_environment.Environment(game)
  learner = mmd_dilated.MMDDilatedEnt(game, alpha=0.1)
                                    
  for ep in range(FLAGS.num_train_episodes):
    learner.update_sequences()
    if (ep + 1) % FLAGS.eval_every == 0:
      expl = exploitability.exploitability(env.game, learner.get_avg_policies())
      logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
    
    if (ep + 1) % FLAGS.save_every == 0:
      with open(FLAGS.checkpoint_dir + f"mmd_dilated_{game_name}_{ep + 1}.pkl", "wb") as f:
        pickle.dump(learner, f)

if __name__ == "__main__":
  app.run(main)
