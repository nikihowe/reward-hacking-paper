# (c) 2022 Nikolaus Howe
import numpy as np

from typing import Callable

from mdp_env import MDPWithoutRewardEnv
from policy import Policy


def ineq_constraints(reward_components,
                     policy_permutation: tuple[Policy],
                     make_reward_fun: Callable,
                     env: MDPWithoutRewardEnv,
                     adjacent_policy_relations: list[int],  # 0: equality, 1: inequality, 2: unspecified
                     ):
    epsilon = 1.

    # Make the reward function using the functional passed in
    reward_fun = make_reward_fun(reward_components)

    # Get the values of the four different policies (in the order they *should* be)
    policy_values = []
    for policy_fun in policy_permutation:
        policy_values.append(env.get_average_policy_value(policy_fun=policy_fun, reward_fun=reward_fun))

    # Get the differences between adjacent policy performances
    ineqs = []
    for i, policy in enumerate(policy_values):
        if i > 0 and adjacent_policy_relations[i - 1] == 1:  # only add in the inequalities
            ineqs.append(policy_values[i] - policy_values[i - 1] - epsilon)

    # If we require positive rewards, enforce that here
    if env.require_nonnegative_reward:
        for reward_component in reward_components:
            ineqs.append(reward_component)

    return ineqs


def make_ineq_constraints(policy_permutation: tuple[Policy],
                          make_reward_fun: Callable,
                          env: MDPWithoutRewardEnv,
                          adjacent_policy_relations: list[int]):
    def curried_ineq_constraints(decision_vars):
        return ineq_constraints(decision_vars, policy_permutation, make_reward_fun, env,
                                adjacent_policy_relations=adjacent_policy_relations)

    return curried_ineq_constraints


def eq_constraints(decision_vars: np.ndarray,
                   policy_permutation: tuple[Policy],
                   env: MDPWithoutRewardEnv,
                   make_reward_fun: Callable,
                   adjacent_policy_relations: list[int]):  # 0: equality, 1: inequality, 2: unspecified
    """
    Make an equality constraint setting the values of the two policies equal
    """
    reward_fun = make_reward_fun(decision_vars)

    # Add equality constraints between the explicitly equated policy pairs
    eq_constraints = []
    # for policy1, policy2 in equal_policy_pairs:
    #     value1 = env.get_average_policy_value(policy_fun=policy1, reward_fun=reward_fun)
    #     value2 = env.get_average_policy_value(policy_fun=policy2, reward_fun=reward_fun)
    #     eq_constraints.append(value2 - value1)

    # Add equality constraints between adjacent policies in the ordering if requested
    for i, e in enumerate(adjacent_policy_relations):
        if e == 0:  # 0: equality, 1: inequality, 2: unspecified
            left_policy, right_policy = policy_permutation[i], policy_permutation[i + 1]
            # print("equating policies", left_policy, right_policy)
            left_value = env.get_average_policy_value(policy_fun=left_policy, reward_fun=reward_fun)
            right_value = env.get_average_policy_value(policy_fun=right_policy, reward_fun=reward_fun)
            eq_constraints.append(right_value - left_value)

    return np.array(eq_constraints)


def make_eq_constraints(env: MDPWithoutRewardEnv,
                        policy_permutation: tuple[Policy],
                        make_reward_fun: Callable,
                        adjacent_policy_relations: list[int]):
    def curried_eq_constraints(decision_vars):
        return eq_constraints(decision_vars=decision_vars,
                              policy_permutation=policy_permutation,
                              env=env,
                              make_reward_fun=make_reward_fun,
                              adjacent_policy_relations=adjacent_policy_relations)

    return curried_eq_constraints


def fancy_print_permutation(policy_permutation, adjacent_policy_relations, realized_rewards=None):
    for i, policy in enumerate(policy_permutation[:-1]):
        relation = '=' if adjacent_policy_relations[i] == 0 else '<' if adjacent_policy_relations[i] == 1 else '?'
        print(policy, relation, end=' ')
    print(policy_permutation[-1], end=' ')
    if realized_rewards is not None:
        print(realized_rewards)
    else:
        print()
