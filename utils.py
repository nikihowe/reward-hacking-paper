# (c) 2022 Nikolaus Howe
import numpy as np

from itertools import combinations
from typing import Callable

from mdp_env import MDPEnv
from policy import Policy


def ineq_constraints(decision_vars,
                     policy_permutation: tuple[Policy],
                     make_reward_fun: Callable,
                     num_eps: int,
                     env: MDPEnv,
                     adjacent_policy_relations: list[int],  # 0: equality, 1: inequality, 2: unspecified
                     ):
    # Extract the current reward decision variables (and current epsilons)
    epss = decision_vars[-num_eps:]

    # Make the reward function using the functional passed in
    reward_fun = make_reward_fun(decision_vars)

    # Get the values of the four different policies (in the order they *should* be)
    policy_values = []
    for policy_fun in policy_permutation:
        policy_values.append(env.get_average_policy_value(policy_fun=policy_fun, reward_fun=reward_fun))

    # Get the differences between adjacent policy performances
    ineqs_with_eps = []
    for i, policy in enumerate(policy_values):
        if i == 0 or adjacent_policy_relations[i - 1] == 0:  # only add in the inequalities
            continue
        ineqs_with_eps.append(policy_values[i] - policy_values[i - 1] - epss[i - 1])

    # Make each epsilon not negative
    for i, e in enumerate(epss):
        if adjacent_policy_relations[i] == 1:
            ineqs_with_eps.append(e - 0.1)  # since reward scale is arbitrary, 0.1 is fine here (want eps >> 0)
        elif adjacent_policy_relations[i] == 0:
            ineqs_with_eps.append(e)  # for some reason, this needs to be handled as ineq. -
            ineqs_with_eps.append(-e)  # mysteriously using eq. doesn't always find a solution
        else:
            ineqs_with_eps.append(e)


    # Make at least one epsilon not negative
    ineqs_with_eps.append(epss[0] + epss[1] + epss[2] - 1e-4)

    return ineqs_with_eps


def make_ineq_constraints(policy_permutation: tuple[Policy],
                          make_reward_fun: Callable,
                          num_eps: int,
                          env: MDPEnv,
                          adjacent_policy_relations: list[int]):
    def curried_ineq_constraints(decision_vars):
        return ineq_constraints(decision_vars, policy_permutation, make_reward_fun, num_eps, env,
                                adjacent_policy_relations=adjacent_policy_relations)

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


def get_specific_eq_constraints(decision_vars: np.ndarray,
                                policy_permutation: list[Policy],
                                env: MDPEnv,
                                make_reward_fun: Callable,
                                equal_policy_pairs: list[tuple[Policy, Policy]],
                                num_eps: int,
                                adjacent_policy_relations: list[int]):  # 0: equality, 1: inequality, 2: unspecified
    """
    Make an equality constraint setting the values of the two policies equal
    """
    assert len(adjacent_policy_relations) == num_eps

    reward_fun = make_reward_fun(decision_vars)
    epss = decision_vars[-num_eps:]

    # Add equality constraints between the explicitly equated policy pairs
    eq_constraints = []
    for policy1, policy2 in equal_policy_pairs:
        value1 = env.get_average_policy_value(policy_fun=policy1, reward_fun=reward_fun)
        value2 = env.get_average_policy_value(policy_fun=policy2, reward_fun=reward_fun)
        eq_constraints.append(value2 - value1)

    # Add equality constraints between adjacent policies in the ordering if requested
    for i, e in enumerate(adjacent_policy_relations):
        if e == 0:  # 0: equality, 1: inequality, 2: unspecified
            left_policy, right_policy = policy_permutation[i], policy_permutation[i + 1]
            # print("equating policies", left_policy, right_policy)
            left_value = env.get_average_policy_value(policy_fun=left_policy, reward_fun=reward_fun)
            right_value = env.get_average_policy_value(policy_fun=right_policy, reward_fun=reward_fun)
            eq_constraints.append(right_value - left_value - epss[i])  # somehow this pushes epss[i] to 0
            # eq_constraints.append(right_value - left_value)  # somehow this pushes epss[i] to 0
            # eq_constraints.append(epss[i] - 1e-8)  # make sure epsilon is 0  # somehow this doesn't work

    return np.array(eq_constraints)


def make_specific_eq_constraints(env: MDPEnv,
                                 policy_permutation: tuple[Policy],
                                 make_reward_fun: Callable,
                                 equal_policy_pairs: list[tuple[Policy, Policy]],
                                 num_eps: int,
                                 adjacent_policy_relations: list[int]):
    def curried_specific_eq_constraints(decision_vars):
        return get_specific_eq_constraints(decision_vars=decision_vars,
                                           policy_permutation=policy_permutation,
                                           env=env,
                                           make_reward_fun=make_reward_fun,
                                           equal_policy_pairs=equal_policy_pairs,
                                           num_eps=num_eps,
                                           adjacent_policy_relations=adjacent_policy_relations)

    return curried_specific_eq_constraints


def check_reward_gameability(holistic_reward_fun, narrow_reward_fun):
    """
    Check if the narrow reward function is gameable with respect to the
    holistic reward function.
    """

    all_actions = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)]

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
