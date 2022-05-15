# (c) 2021 Nikolaus Howe
import numpy as np

from itertools import combinations
from typing import Callable

from mdp_env import MDPEnv
from policy import Policy


def ineq_constraints(decision_vars, policy_permutation: list[Policy], make_reward_fun: Callable, env: MDPEnv):
    # Extract the current reward decision variables (and current epsilons)
    r00, r01, r10, r11, *epss = decision_vars

    # Make the reward function using the functional passed in
    reward_fun = make_reward_fun(decision_vars)

    # Get the values of the four different policies (in the order they *should* be)
    policy_values = []
    for policy_fun in policy_permutation:
        policy_values.append(env.get_average_policy_value(policy_fun=policy_fun, reward_fun=reward_fun))

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


def make_ineq_constraints(policy_permutation, make_reward_fun, env):
    def curried_ineq_constraints(decision_vars):
        return ineq_constraints(decision_vars, policy_permutation, make_reward_fun, env)

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


def get_specific_eq_constraints(decision_vars, env: MDPEnv, make_reward_fun, worse_policy, better_policy):
    """
    Make an equality constraint setting the values of the two policies equal
    """
    # rewards = decision_vars[:4].reshape((2, 2))
    reward_fun = make_reward_fun(decision_vars)

    v_worse_policy = env.get_average_policy_value(worse_policy, reward_fun=reward_fun)
    v_better_policy = env.get_average_policy_value(better_policy, reward_fun=reward_fun)
    return np.array([v_worse_policy - v_better_policy])


def make_specific_eq_constraints(env, make_reward_fun, worse_policy, better_policy):
    def curried_specific_eq_constraints(decision_vars):
        return get_specific_eq_constraints(decision_vars=decision_vars,
                                           env=env,
                                           make_reward_fun=make_reward_fun,
                                           worse_policy=worse_policy,
                                           better_policy=better_policy)

    return curried_specific_eq_constraints


# TODO: is it enough to consider setting exactly two policies equal, or must we
#  also consider when two policies are equal and the last one isn't

def check_reward_gameability(holistic_reward_fun, narrow_reward_fun):
    """
    Check if the narrow reward function is gameable with respect to the
    holistic reward function.
    """

    all_actions = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]]
    )

    counter = 0
    for action1, action2 in combinations(all_actions, 2):
        counter += 1
        if holistic_reward_fun(action1) < holistic_reward_fun(action2) and \
                narrow_reward_fun(action2) < narrow_reward_fun(action1):
            print(f"The holistic reward function says that {action2}: "
                  f"{holistic_reward_fun(action2)} is better than {action1}: {holistic_reward_fun(action1)}")
            print(f"but the narrow reward function says that {action1}: "
                  f"{holistic_reward_fun(action1)} is better than {action2}: {narrow_reward_fun(action2)}")
            return False

    print("checked", counter)
    return True


if __name__ == "__main__":
    def hr1(action):
        return action @ np.array([3, 4, 5])

    def nr1(action):
        return action @ np.array([4, 4, 4])

    def hr2(action):
        return action @ np.array([5, 2, 2])

    def nr2(action):
        return action @ np.array([1, 0, 0])

    def nr3(action):
        return action @ np.array([2, 2, 2])

    def hr3(action):
        return action @ np.array([1, 0, 0])


    print(check_reward_gameability(hr1, nr1))
    print(check_reward_gameability(hr2, nr2))
    print(check_reward_gameability(hr3, nr3))

