# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from mdp_env import MDPEnv
from permutations import calculate_achievable_permutations
from policy import Policy, make_two_state_policy
import utils


def run_simplification_search(adjacent_policy_relations: list[int],
                              equal_policy_list: list[tuple[Policy, Policy]],
                              policy_permutations: list[tuple[Policy]],
                              make_reward_fun: Callable,
                              run_single_env_specific_search: Callable,
                              env: MDPEnv) -> None:
    print("policy funs being equated:", equal_policy_list)
    num_eps = len(policy_permutations[0]) - 1
    print("num_eps:", num_eps)

    for policy_permutation in policy_permutations:
        print(f"Given permutation: {policy_permutation}")
        print(f"Trying to set equal {equal_policy_list}")
        print(f"Adjacent policy relations: {adjacent_policy_relations}")

        eq_constraints = utils.make_specific_eq_constraints(env=env,
                                                            policy_permutation=policy_permutation,
                                                            make_reward_fun=make_reward_fun,
                                                            equal_policy_pairs=equal_policy_list,
                                                            num_eps=num_eps,
                                                            adjacent_policy_relations=adjacent_policy_relations)
        ineq_constraints = utils.make_ineq_constraints(adjacent_policy_relations=adjacent_policy_relations,
                                                       policy_permutation=policy_permutation,
                                                       make_reward_fun=make_reward_fun,
                                                       num_eps=num_eps,
                                                       env=env)
        run_single_env_specific_search(eq_constraints=eq_constraints,
                                       ineq_constraints=ineq_constraints,
                                       num_eps=num_eps,
                                       make_reward_fun=make_reward_fun,
                                       policy_permutation=policy_permutation,
                                       env=env)

        print("#######################################################")
