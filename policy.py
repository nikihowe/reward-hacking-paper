# (c) 2022 Nikolaus Howe
import dataclasses
import numpy as np

from typing import Any, Callable, Union


@dataclasses.dataclass
class Policy(object):
    """
    A policy is a function that maps a state to an action.
    """
    name: Any
    policy_fun: Callable[[int], Union[int, tuple[int, int, int]]]

    def __call__(self, state: int) -> Union[int, np.ndarray]:
        return self.policy_fun(state)

    def __repr__(self):
        return f"Policy{str(self.name)}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return str(self.name) == str(other.name)

    def __le__(self, other):
        return str(self.name) <= str(other.name)

    def __lt__(self, other):
        return str(self.name) < str(other.name)


def make_two_state_policy(policy_tuple: tuple[int, int]) -> Policy:
    return Policy(policy_tuple, lambda state: policy_tuple[state])


def make_cleaning_policy(policy_tuple: tuple[int, int, int]) -> Policy:
    return Policy(policy_tuple, lambda state: policy_tuple)  # note that we don't care about state in cleaning robot
