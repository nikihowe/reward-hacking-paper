# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from mdp_env import MDPEnv
from policy import Policy
import utils


# Calculate which permutations are possible
def calculate_achievable_permutations(allowed_policies: list[Policy],
                                      make_reward_fun: Callable,
                                      env: MDPEnv,
                                      reward_size: int,
                                      search_steps: int,
                                      show_rewards: bool = False,
                                      print_output: bool = False) -> list[tuple[Policy]]:
    print("Considering all orderings of the following policies:")
    print(allowed_policies, '\n--------------------------------\n')

    num_eps = len(allowed_policies) - 1

    all_permutations = list(itertools.permutations(allowed_policies))
    all_relations = [(i, j, k) for i, j, k in itertools.product(range(2), range(2), range(2))]

    weak_orderings = []
    for perm in all_permutations:
        for relation in all_relations:
            weak_orderings.append((perm, relation))

    done = []
    for i, (perm, relation) in enumerate(weak_orderings):
        if i % 10 == 0:
            print(f"testing {i+1} of {len(weak_orderings)}")
        eq_constraints = utils.make_eq_constraints(env=env,
                                                   policy_permutation=perm,
                                                   make_reward_fun=make_reward_fun,
                                                   equal_policy_pairs=[],
                                                   num_eps=num_eps,
                                                   adjacent_policy_relations=list(relation))
        ineq_constraints = utils.make_ineq_constraints(adjacent_policy_relations=list(relation),
                                                       policy_permutation=perm,
                                                       make_reward_fun=make_reward_fun,
                                                       num_eps=num_eps,
                                                       env=env)
        res = minimize(
            fun=lambda x: 0,
            x0=np.ones(reward_size + num_eps),
            constraints=
            [{"type": "eq",
              "fun": eq_constraints},
             {"type": "ineq",
              "fun": ineq_constraints}]
        )
        if res.success:
            done.append((perm, relation))

    print(done)

    # for _ in range(search_steps):
    #     dec_vars = np.random.uniform(0, 1, reward_size + num_eps)  # rewards and epsilons
    #
    #     # Check whether we've already seen this permutation
    #     temp_reward_fun = make_reward_fun(dec_vars[:-num_eps])
    #     policies_and_rewards = env.get_sorted_policies_and_rewards(allowed_policies, reward_fun=temp_reward_fun)
    #
    #     # If we've seen this permutation, skip it
    #     policy_permutation = []
    #     for (p, r) in policies_and_rewards:
    #         policy_permutation.append(p)
    #     policy_permutation = tuple(policy_permutation)
    #     if policy_permutation in all_permutations:
    #         if print_output:
    #             print(f"Possible ordering: {policy_permutation}")
    #             if show_rewards:
    #                 print(f"Achieved with rewards: {dec_vars[:-num_eps]}")
    #         all_permutations.remove(policy_permutation)
    #         considered_permutations.append(policy_permutation)
    #         print()
    #     else:
    #         continue
    #
    # print("there were {} achieved permutations".format(len(considered_permutations)))
    # print("there were {} not achieved permutations".format(len(all_permutations)))
    # print("\n\n")
    #
    # return considered_permutations
