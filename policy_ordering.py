# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from environment import MDPWithoutRewardEnv
from policy import Policy
import constraints


def _policy_ordering_search_solver(eq_constraints: Callable,
                                   ineq_constraints: Callable,
                                   make_reward_fun: Callable,
                                   reward_size: int,
                                   policy_permutation: tuple[Policy],
                                   env: MDPWithoutRewardEnv) -> bool:
    res = minimize(
        fun=lambda x: 0,
        x0=np.ones(reward_size),
        constraints=
        [{"type": "eq",
          "fun": eq_constraints},
         {"type": "ineq",
          "fun": ineq_constraints}]
    )
    if res.success:
        print("Success! The values of the policies are")
        temp_rf = make_reward_fun(res.x)
        all_ave_policy_vals = env.get_all_average_policy_values(policy_permutation=policy_permutation,
                                                                reward_fun=temp_rf)
        for i, policy in enumerate(policy_permutation):
            print(f"{policy}: {round(all_ave_policy_vals[i], 2)}")
        print()

        print("Using rewards")
        for i in range(reward_size):
            print(f"{i}: {round(res.x[i], 2)}")
        print()
        print()
    else:
        print("Unable to find a reward function that achieves this ordering.")
        print()

    return res.success


# Given a policy permutation and adjacent policy relations, try to find a reward
# function which satisfies them.
def run_policy_ordering_search(policy_permutation, adjacent_relations, make_reward_fun, reward_size, env) -> bool:
    eq_constraints = constraints.make_eq_constraints(env=env,
                                                     policy_permutation=policy_permutation,
                                                     make_reward_fun=make_reward_fun,
                                                     adjacent_policy_relations=adjacent_relations)
    ineq_constraints = constraints.make_ineq_constraints(adjacent_policy_relations=adjacent_relations,
                                                         policy_permutation=policy_permutation,
                                                         make_reward_fun=make_reward_fun,
                                                         env=env)
    success = _policy_ordering_search_solver(eq_constraints=eq_constraints,
                                             ineq_constraints=ineq_constraints,
                                             make_reward_fun=make_reward_fun,
                                             reward_size=reward_size,
                                             policy_permutation=policy_permutation,
                                             env=env)

    return success


# Given a policy permutation, test all adjacent policy relations to see if
# there is a reward function which achieves them.
def run_adjacent_relation_search(policy_permutation: tuple[Policy],
                                 make_reward_fun: Callable,
                                 reward_size: int,
                                 env: MDPWithoutRewardEnv) -> list[tuple[int]]:
    list_of_all_adjacent_relations = list(itertools.product(*([range(2)] * (len(policy_permutation) - 1))))

    successful_relations = []
    for adjacent_relations in list_of_all_adjacent_relations:
        print(f"Permutation: {policy_permutation}")
        print(f"Adjacent policy relations: {adjacent_relations}")

        success = run_policy_ordering_search(policy_permutation, adjacent_relations, make_reward_fun, reward_size, env)
        if success:
            successful_relations.append(adjacent_relations)

    return successful_relations


# Test all policy orderings and all adjacent policy relations to see if there is a
# reward function which achieves them.
def run_full_ordering_search(policies: list[Policy],
                             make_reward_fun: Callable,
                             reward_size: int,
                             env: MDPWithoutRewardEnv) -> list[tuple[tuple[Policy], tuple[int]]]:
    successful_orderings_with_relations = []
    for policy_permutation in itertools.permutations(policies):
        # print("Now considering policy permutation:", policy_permutation)
        successful_relations = run_adjacent_relation_search(policy_permutation, make_reward_fun, reward_size, env)
        for successful_relation in successful_relations:
            successful_orderings_with_relations.append((policy_permutation, successful_relation))
        # successful_orderings_with_relations.append((policy_permutation, successful_relations))

    return successful_orderings_with_relations

# def run_full_simplification_search(adjacent_policy_relations: list[int],
#                                    policy_permutations: list[tuple[Policy]],
#                                    make_reward_fun: Callable,
#                                    reward_size: int,
#                                    env: MDPWithoutRewardEnv) -> None:
#     num_eps = len(policy_permutations[0]) - 1
#     print("num_eps:", num_eps)
#
#     for policy_permutation in policy_permutations:
#         print(f"Given permutation: {policy_permutation}")
#         print(f"Adjacent policy relations: {adjacent_policy_relations}")
#
#         eq_constraints = utils.make_eq_constraints(env=env,
#                                                    policy_permutation=policy_permutation,
#                                                    make_reward_fun=make_reward_fun,
#                                                    adjacent_policy_relations=adjacent_policy_relations)
#         ineq_constraints = utils.make_ineq_constraints(adjacent_policy_relations=adjacent_policy_relations,
#                                                        policy_permutation=policy_permutation,
#                                                        make_reward_fun=make_reward_fun,
#                                                        env=env)
#         run_single_simplification_search(eq_constraints=eq_constraints,
#                                          ineq_constraints=ineq_constraints,
#                                          num_eps=num_eps,
#                                          make_reward_fun=make_reward_fun,
#                                          reward_size=reward_size,
#                                          policy_permutation=policy_permutation,
#                                          env=env)
#
#         print("#######################################################")
