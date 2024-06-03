
#imports
from typing import Callable, Optional, Protocol, Union
import pandas as pd
import pickle
from absl import app
from absl import flags
from absl import logging

import open_spiel.python.games
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import mmd_dilated
from open_spiel.python.algorithms import cfr
import pyspiel

FLAGS = flags.FLAGS

# flags 
flags.DEFINE_string("game", "python_block_dominoes", "Name of the game.")
flags.DEFINE_integer("num_train_episodes", 20_000,
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 200,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("save_every", 5000,
                     "Episode frequency at which the agents are saved.")
flags.DEFINE_string("checkpoint_dir", "open_spiel/python/examples/tiny_block_dominoes/agents/", 
                    "Directory to save/load the agent models.")
flags.DEFINE_string("results_dir", "open_spiel/python/examples/tiny_block_dominoes/results/train/", 
                    "Directory to save the data.")



# hyperparameters
flags.DEFINE_float("alpha", 0.05, "Learning rate for the MMD model.")

# main loop



def main(unused_argv):
    game_name = FLAGS.game
    game = pyspiel.load_game(FLAGS.game)
    learners = {"cfr": cfr.CFRSolver(game),
                "cfrplus": cfr.CFRPlusSolver(game),
                "mmd": mmd_dilated.MMDModel(game, FLAGS.alpha)
    }
    for alg_name, learner in learners.items():
      df = pd.DataFrame({})
      # train                                 
      for ep in range(FLAGS.num_train_episodes):
          if alg_name == "cfr" or alg_name == "cfrplus":
            learner.evaluate_and_update_policy()
          else:
            learner.update_sequences()
          if (ep + 1) % FLAGS.eval_every == 0:
              if alg_name == "cfr" or alg_name == "cfrplus":
                expl = exploitability.exploitability(game, learner.average_policy())
              else:
                expl = exploitability.exploitability(game, learner.get_avg_policies())
              # save exploitability in csv
              df = pd.concat([df, pd.DataFrame(log_info(ep, expl))], ignore_index=True)
              df.to_csv(FLAGS.results_dir + f"{alg_name}_{game_name}_{FLAGS.num_train_episodes}.csv")
            
          if (ep + 1) % FLAGS.save_every == 0:
            with open(FLAGS.checkpoint_dir + f"{alg_name}_{game_name}_{ep + 1}.pkl", "wb") as f:
              pickle.dump(learner, f)
              logging.info(f"[{ep + 1}] Saved {alg_name} learner")



# auxiliary functions

# plot it and save it
def log_info(ep, expl) -> dict[str, list[Union[float, str]]]:
  logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
  return {
    "Iteration": [ep+1],
    "Exploitability": [expl]
  }
  

# run the main loop
if __name__ == "__main__":
    app.run(main)
