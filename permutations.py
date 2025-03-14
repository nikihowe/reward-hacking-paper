# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from environment import MDPWithoutRewardEnv
from policy import Policy
import constraints


# Calculate which permutations are possible
def calculate_achievable_permutations(allowed_policies: list[Policy],
                                      make_reward_fun: Callable,
                                      env: MDPWithoutRewardEnv,
                                      reward_size: int,
                                      ):
    print("Considering all orderings of the following policies:")
    print(allowed_policies, '\n--------------------------------\n')

    all_permutations = list(itertools.permutations(allowed_policies))
    all_relations = list(itertools.product(*([range(2)] * (len(allowed_policies) - 1))))

    weak_orderings = []
    for perm in all_permutations:
        for relation in all_relations:
            weak_orderings.append((perm, relation))

    realized_permutations = []
    realized_relations = []
    realized_rewards = []
    for i, (perm, relation) in enumerate(weak_orderings):
        if i % 10 == 0:
            print(f"Working on permutation {i+1} of {len(weak_orderings)}")
        eq_constraints = constraints.make_eq_constraints(env=env,
                                                   policy_permutation=perm,
                                                   make_reward_fun=make_reward_fun,
                                                   adjacent_policy_relations=list(relation))
        ineq_constraints = constraints.make_ineq_constraints(adjacent_policy_relations=list(relation),
                                                       policy_permutation=perm,
                                                       make_reward_fun=make_reward_fun,
                                                       env=env)
        res = minimize(
            fun=lambda x: 0,
            x0=np.zeros(reward_size),
            constraints=
            [{"type": "eq",
              "fun": eq_constraints},
             {"type": "ineq",
              "fun": ineq_constraints}]
        )
        if res.success:
            realized_permutations.append(perm)
            realized_relations.append(relation)
            realized_rewards.append(res.x)

    # for i, perm in enumerate(realized_permutations):
        # utils.fancy_print_permutation(perm, realized_relations[i], realized_rewards[i])
        # print(perm, realized_relations[i], realized_rewards[i])

    return realized_permutations, realized_relations, realized_rewards
