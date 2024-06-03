# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
# Copyright 2021 Artificial Intelligence Center, Czech Techical University
# Copied and adapted from OpenSpiel (https://github.com/deepmind/open_spiel)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import six # TODO: check what this is
from typing import Union
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import time

import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import open_spiel.python.games
import pyspiel


FLAGS = flags.FLAGS

flags.DEFINE_string("game", "python_block_dominoes", "Name of the game")
flags.DEFINE_integer("num_players", 2, "Number of players") # TODO: change to parameters
flags.DEFINE_string("project", "openspiel", "project name")
# flags.DEFINE_bool("tabular", False, "Game is tabular")
flags.DEFINE_bool("verbose", False, "Verbose logging")
flags.DEFINE_integer("eval_every", 1,
    "Episode frequency at which the DQN agents are evaluated.")
flags.DEFINE_integer("save_every", 1, 
                     "Episode frequency at which the agents are saved.") # never for now
flags.DEFINE_string("checkpoint_dir", "/tmp/", 
                    "Directory to save/load the agent models.")
flags.DEFINE_string("results_dir", "open_spiel/python/examples/tiny_block_dominoes/results/train/", 
                    "Directory to save the data.")
flags.DEFINE_bool("use_checkpoints", True, "load neural network weights.")


# training parameters
flags.DEFINE_integer("iterations", 200, "Number of training iterations.")
flags.DEFINE_integer("num_traversals", 1_500, "Number of traversals/games") # used to be 40
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


def main(argv):
    logging.info("Loading %s", FLAGS.game)
    game_name = FLAGS.game
    num_players = FLAGS.num_players
    alg_name = "deepcfr"
    df = pd.DataFrame({})
    rewards = []

    game = pyspiel.load_game(
        FLAGS.game) # {"players": pyspiel.GameParameter(FLAGS.players)}) # {"players": 2}

    # add a lists to store rewards and iteration times
    rewards = []
    iteration_times = []

    with tf.Session() as sess:
        agent = deep_cfr.DeepCFRSolver(
            sess,
            game,
            policy_network_layers=tuple(
                [FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            advantage_network_layers=tuple(
                [FLAGS.num_hidden for _ in range(FLAGS.num_layers)]),
            num_iterations=FLAGS.iterations,
            num_traversals=FLAGS.num_traversals,
            learning_rate=FLAGS.learning_rate,
            batch_size_advantage=FLAGS.batch_size_advantage,
            batch_size_strategy=FLAGS.batch_size_strategy,
            memory_capacity=FLAGS.memory_capacity,
            policy_network_train_steps=FLAGS.policy_network_train_steps,
            advantage_network_train_steps=FLAGS.advantage_network_train_steps,
            reinitialize_advantage_networks=FLAGS.reinitialize_advantage_networks)
        sess.run(tf.global_variables_initializer())

        # if FLAGS.use_checkpoints:
        #     if agent.has_checkpoint(FLAGS.checkpoint_dir):
        #         agent.restore(FLAGS.checkpoint_dir)
        #         logging.info("Loaded checkpoint from '%s'", FLAGS.checkpoint_dir)
        #     else:
        #         logging.info("No checkpoint found in '%s'", FLAGS.checkpoint_dir)


        bots = [
            {'type': 'random', 'bot': None},
            {'type': 'cfr', 'bot': agent}
        ]

        """Modified deep-cfr solution logic for online policy evaluation"""
        advantage_losses = collections.defaultdict(list)

        for ep in range(agent._num_iterations):
            start_time = time.time()
            if ep % FLAGS.eval_every == 0: # evaluation rounds
                print(f"Evaluation round {ep}: evaluating against random bots")
                r_mean = eval_against_random_bots(game, bots, 1000)
                rewards.append(r_mean)
                logging.info("[%s] Mean episode rewards %s", ep, r_mean)
                plot_rewards(rewards)

                # save the iteration time
                iteration_time = time.time() - start_time
                iteration_times.append(iteration_time)
                plot_iteration_times(iteration_times)

                # save the results
                df = pd.concat([df, pd.DataFrame(log_info(ep, r_mean, iteration_time))], ignore_index=True)
                df.to_csv(FLAGS.results_dir + f"{alg_name}_{game_name}_{FLAGS.iterations}_31-05.csv")
            
            # if (ep + 1) % FLAGS.save_every == 0:
            #     agent.save(FLAGS.checkpoint_dir + f"{alg_name}_{game_name}")
            #     logging.info(f"[{ep + 1}] Saved {alg_name} learner")

            for p in range(agent._num_players):
                for _ in range(agent._num_traversals):
                    agent._traverse_game_tree(
                        agent._root_node, p)
                if agent._reinitialize_advantage_networks:
                    # Re-initialize advantage network for player and train from scratch.
                    agent.reinitialize_advantage_network(p)
                advantage_losses[p].append(
                    agent._learn_advantage_network(p))

            # Train policy network.
            policy_loss = agent._learn_strategy_network()
            agent._iteration += 1

# helper functions

# plot it and save it
def log_info(ep, r_mean, iteration_time) -> dict[str, list[Union[float, str]]]:
  return {
    "Iteration": [ep+1],
    "Rewards": [r_mean],
    "Time": [iteration_time]
  }
  

def eval_against_random_bots(game, bots, num_episodes):
    """Evaluates `trained` cfr agent against `random agents`"""
    num_players = len(bots)
    sum_episode_rewards = np.zeros(num_players)
    for episode in range(num_episodes):
        bots, cfr_player_index = alternate_starting_player(bots, episode)
        # print(episode, bots, cfr_player_index)
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                _apply_chance_node_action(state)
            else:
                # state = apply_player_action(state, bots)
                player = state.current_player()
                curr_bot = bots[player]  # Use the current player to select the bot
                action = _get_bot_action(state, curr_bot)
                state.apply_action(action)
                if FLAGS.verbose:
                    print(f"Player {curr_bot['type']:<10} [{player:<2}] chose action: {action:<5} ({state.action_to_string(action):<20})")
        sum_episode_rewards[cfr_player_index] += state.returns()[cfr_player_index]  # Use the current player to update the rewards
        if FLAGS.verbose:
            _print_game_over_info(state)
    return sum_episode_rewards / num_episodes
 

def alternate_starting_player(bots, episode):
    """Alternate the starting player"""
    bots = bots[::-1]
    cfr_player_index = next(i for i, bot in enumerate(bots) if bot['type'] == 'cfr')
    return bots, cfr_player_index 

def _apply_chance_node_action(state):
    outcomes = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes)
    outcome = np.random.choice(action_list, p=prob_list)
    state.apply_action(outcome)

def _get_bot_action(state, curr_bot):
    """Get action based on bot type"""
    if curr_bot['type'] == 'random':
        return np.random.choice(state.legal_actions())
    elif curr_bot['type'] == 'cfr':
        action_probs = curr_bot['bot'].action_probabilities(state)
        action = max(action_probs, key=action_probs.get)

        if FLAGS.verbose:
            _print_bot_action_probabilities(curr_bot, state, action_probs)
        return action

def _print_bot_action_probabilities(curr_bot, state, action_probs):
    """Print bot action probabilities"""
    print(f"--- {curr_bot['type']} action probabilities ----")
    for action, probability in action_probs.items():
        print(f"    Action: {state.action_to_string(action)}, Probability: {probability}")
    print(f"--- ------------------- ----")

def _print_game_over_info(state):
    """Print game over information"""
    print("\n-=- Game over -=-\n")
    print(f"Terminal state:\n{state}")
    print(f"Returns: {state.returns()}")

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
        

def plot_iteration_times(iteration_times):
    fig = plt.figure()
    plt.plot(iteration_times, color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title(f'{FLAGS.game}: Time per Iteration')
    plt.savefig(f'open_spiel/python/examples/saved_examples/{FLAGS.game}_time_plot_deepcfr-08-05.png')



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



if __name__ == "__main__":
    app.run(main)