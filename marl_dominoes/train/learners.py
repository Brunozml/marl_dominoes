"""Learners for small extensive-form games using OpenSpiel. 
Inspired by [mmd code](https://github.com/ssokota/mmd/blob/master/mmd/efg/learners.py)
"""

from collections import defaultdict
from math import prod
from typing import Callable, Optional, Protocol, Union

import numpy as np
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms.cfr import _CFRSolver
from pyspiel import Game


class Learner(Protocol):
    def update(self) -> None:
        """Perform update for policies, increment `iteration` by one"""

    def test_policy(self) -> TabularPolicy: # probably need to change
        """Return test policy"""

    def log_info(self) -> dict[str, list[Union[float, str]]]:
        """Return relevant learning information"""

    @property
    def game(self) -> Game:
        """Return the game for learning"""

    @property
    def comparator(self) -> Optional[np.ndarray]:
        """Return the policy from which to compute KL divergence"""

    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""
    

class CFR(Learner):
    def __init__(self, game: Game, use_plus: bool):
        self._game = game
        self.solver = _CFRSolver(
            game,
            regret_matching_plus=use_plus,
            alternating_updates=use_plus,
            linear_averaging=use_plus,
        )
        self._comparator = None
        self._iteration = 0

    def update(self) -> None:
        """Perform update for policies, increment `iteration` by one"""
        self.solver.evaluate_and_update_policy()
        self._iteration += 1

    def test_policy(self) -> np.ndarray:
        """Return test policy"""
        return self.solver.average_policy()

    def log_info(self) -> dict[str, list[Union[float, str]]]:
        """Return relevant learning information"""
        return {
            "Iteration": [self.iteration],
            "Exploitability": [exploitability(self.game, self.test_policy())],
        }

    @property
    def game(self) -> Game:
        """Return the game for learning"""
        return self._game

    @property
    def comparator(self) -> Optional[np.ndarray]:
        """Return the policy from which to compute KL divergence"""
        return self._comparator
    
    @property
    def iteration(self) -> int:
        """Return the number of updates that have been performed"""
        return self._iteration
    

