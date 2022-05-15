# (c) 2021 Nikolaus Howe
import numpy as np

from itertools import combinations

from mdp_env import MDPEnv


def ineq_constraints(decision_vars, policy_permutation, env: MDPEnv):
    # Extract the current reward decision variables (and current epsilons)
    r00, r01, r10, r11, *epss = decision_vars

    # Reshape them into rewards
    rewards = decision_vars[:4].reshape((2, 2))

    # Get the values of the four different policies (in the order they *should* be)
    policy_values = []
    for policy in policy_permutation:
        policy_values.append(env.get_average_policy_value(policy, rewards))

    # Get the differences between adjacent policy performances
    ineqs_with_eps = []
    for i, policy in enumerate(policy_values):
        if i == 0:
            continue
        ineqs_with_eps.append(policy_values[i] - policy_values[i - 1] - epss[i - 1])

    # Make each epsilon not negative
    for e in epss:
        ineqs_with_eps.append(e - 1e-8)

    # Make at least one epsilon not negative
    ineqs_with_eps.append(epss[0] + epss[1] + epss[2] - 1e-4)

    return ineqs_with_eps


def make_ineq_constraints(policy_permutation, env):
    def curried_ineq_constraints(decision_vars):
        return ineq_constraints(decision_vars, policy_permutation, env)

    return curried_ineq_constraints


# def eq_constraints(decision_vars, env: MDPEnv):
#     rewards = decision_vars[:4].reshape((2, 2))
#     q00 = env.get_Q(state=0, action=0, rewards=rewards)
#     q01 = env.get_Q(state=0, action=1, rewards=rewards)
#     #   print("q00, q01", q00, q01)
#     return np.array([q01 - q00])


# def make_eq_constraints(env):
#     def curried_eq_constraints(decision_vars):
#         return eq_constraints(decision_vars, env)
#
#     return curried_eq_constraints


def get_specific_eq_constraints(decision_vars, env: MDPEnv, worse_policy, better_policy):
    """
    Make an equality constraint setting the values of the two policies equal
    """
    rewards = decision_vars[:4].reshape((2, 2))
    v_worse_policy = env.get_average_policy_value(worse_policy, rewards)
    v_better_policy = env.get_average_policy_value(better_policy, rewards)
    return np.array([v_worse_policy - v_better_policy])


def make_specific_eq_constraints(env, worse_policy, better_policy):
    def curried_specific_eq_constraints(decision_vars):
        return get_specific_eq_constraints(decision_vars, env, worse_policy, better_policy)

    return curried_specific_eq_constraints

# TODO: is it enough to consider setting exactly two policies equal, or must we
#  also consider when two policies are equal and the last one isn't