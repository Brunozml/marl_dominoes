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
flags.DEFINE_integer("num_train_episodes", 10000,
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 50,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 100,
                     "Episode frequency at which the agents are saved.")
flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/saved_examples/tiny_block_dominoes/agents/", 
                    "Directory to save/load the agent models.")
flags.DEFINE_integer("last_checkpoint", 100, "Last checkpoint to load.")


def main(unused_argv):
  game_name = FLAGS.game
  game = pyspiel.load_game(game_name)
  env = rl_environment.Environment(game)
  # Load the learner
  with open(FLAGS.checkpoint_dir + f"mmd_dilated_{game_name}_{FLAGS.last_checkpoint}.pkl", 'rb') as input:
      learner = pickle.load(input)
      logging.info("Loaded learner from %s", FLAGS.checkpoint_dir + f"mmd_dilated_{game_name}_{FLAGS.last_checkpoint}.pkl")
                                    
  for ep in range(FLAGS.last_checkpoint, FLAGS.num_train_episodes):
    learner.update_sequences()
    if (ep + 1) % FLAGS.eval_every == 0:
      expl = exploitability.exploitability(env.game, learner.get_avg_policies())
      logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
    
    if (ep + 1) % FLAGS.save_every == 0:
      with open(FLAGS.checkpoint_dir + f"mmd_dilated_{game_name}_{ep + 1}.pkl", "wb") as f:
        pickle.dump(learner, f)

if __name__ == "__main__":
  app.run(main)
