import argparse
import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import torch
import jax

import pyspiel
import open_spiel.python.games


game_name = "python_block_dominoes"
results_dir = "open_spiel/python/examples/tiny_block_dominoes/results/h2h/"
agents_dir = "open_spiel/python/examples/tiny_block_dominoes/agents/"
agent_choices = ["cfr", "cfrplus", "mmd", "random", "qlearner"]
num_episodes = 1_000
seeds = [0, 1, 2]

def make_player(dir: str, game: str, agent: str, seed: str):
    # specify process for cfr and cfrplus
    if agent in ["cfr", "cfrplus"]:
        with open (f"{dir}/{agent}_{game}_20000.pkl", 'rb') as f:
            learner = pickle.load(f)
        policy = learner.average_policy()
        return player_factory(policy)
    elif agent == "random":
        return random
    elif agent == "mmd":
        with open (f"{dir}/{agent}_dilated_{game}_1000.pkl", 'rb') as f:
            learner = pickle.load(f)
        policy = learner.get_avg_policies()
        return player_factory(policy)
    elif agent == "qlearner":
        with open (f"{dir}/{agent}_{game}_3000000.pkl", 'rb') as f:
            policy = pickle.load(f)
        return player_factory(policy)
    else:
        raise ValueError(f"Unknown agent: {agent}")


def random(state: pyspiel.State) -> int:
    return np.random.choice(state.legal_actions())


def player_factory(policy):
    def player(state: pyspiel.State) -> int:
        action_probs = policy.action_probabilities(state)
        action_list = list(action_probs.keys())
        action = np.random.choice(action_list, p=list(action_probs.values()))
        return action
    return player


def matchup(game: pyspiel.Game,
            players: list[Callable[[list[float], list[int]], int]],
            num_episodes: int,
) -> None:
    """Matchup players in a game for a number of episodes."""
    results = []
    for i in range(num_episodes):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
            else: 
                player = players[state.current_player()]
                action = player(state)
        
            state.apply_action(action)
        # record results
        results.append(state.returns())
        # save results
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent1",
        choices=agent_choices,
        required=True,
    )
    parser.add_argument(
        "--agent2",
        choices=agent_choices,
        required=True,
    )
    args = parser.parse_args()
    game = pyspiel.load_game(game_name)
    agents = [
        [make_player(agents_dir, game_name, agent, s) for s in seeds]
        for agent in [args.agent1, args.agent2]
    ]
    colname = f"{args.agent1} Return Against {args.agent2}"
    df = pd.DataFrame(columns=  [colname, "Order"])

    for s1, a1 in enumerate(agents[0]):
        for s2, a2 in enumerate(agents[1]):
            # agent 1 moves first
            results = matchup(game, [a1, a2], num_episodes)
            # Append results to the DataFrame
            for result in results:
                df = pd.concat([df, pd.DataFrame({colname: [result[0]], "Order": [0]})], ignore_index=True)

            # agent 2 moves first
            results = matchup(game, [a2, a1], num_episodes)
            # Append results to the DataFrame
            for result in results:
                df = pd.concat([df, pd.DataFrame({colname: [result[1]], "Order": [1]})], ignore_index=True)



    # save
    df.to_csv(results_dir + f"{game_name}_{args.agent1}_{args.agent2}.csv")
    # plot
    sns.boxplot(data=df, x="Order", y = colname)
    plt.savefig(results_dir + f"/{game_name}_{args.agent1}_{args.agent2}.png")
    expected_return = round(df[colname].mean(), 2)
    std_err = round(df[colname].std() / np.sqrt(len(df[colname])), 2)
    print(f"Expected {colname}: {expected_return} +/- {std_err}")


from open_spiel.python import policy
from open_spiel.python import rl_environment
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


if __name__ == "__main__":
    main()