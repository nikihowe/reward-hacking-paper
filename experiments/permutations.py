# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from typing import Callable

from mdp_env import MDPEnv
from policy import Policy


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

    considered_permutations = []

    for _ in range(search_steps):
        dec_vars = np.random.uniform(0, 1, reward_size + num_eps)  # rewards and epsilons

        # Check whether we've already seen this permutation
        temp_reward_fun = make_reward_fun(dec_vars[:-num_eps])
        policies_and_rewards = env.get_sorted_policies_and_rewards(allowed_policies, reward_fun=temp_reward_fun)

        # If we've seen this permutation, skip it
        policy_permutation = []
        for (p, r) in policies_and_rewards:
            policy_permutation.append(p)
        policy_permutation = tuple(policy_permutation)
        if policy_permutation in all_permutations:
            if print_output:
                print(f"Possible ordering: {policy_permutation}")
                if show_rewards:
                    print(f"Achieved with rewards: {dec_vars[:-num_eps]}")
            all_permutations.remove(policy_permutation)
            considered_permutations.append(policy_permutation)
            print()
        else:
            continue

    print("there were {} achieved permutations".format(len(considered_permutations)))
    print("there were {} not achieved permutations".format(len(all_permutations)))
    print("\n\n")

    return considered_permutations
