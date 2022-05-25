# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from mdp_env import MDPEnv
from permutations import calculate_achievable_permutations
from policy import Policy, make_two_state_policy
from simplification import run_simplification_search
import utils

REWARD_SIZE = 4  # four (s, a) pairs, different reward for each
REWARD_SHAPE = (2, 2)
SEARCH_STEPS = 200


#################
# Two-state MDP #
#################

# It's a two-state two-action environment, with deterministic dynamics
def dynamics(state, action):
    del state
    return action


# Reward is deterministic and depends on state and action
def make_reward_fun_from_dec_vars(dec_vars):
    rewards = dec_vars[:REWARD_SIZE].reshape(REWARD_SHAPE)

    def reward_fun(state, action):
        return rewards[state, action]

    return reward_fun


def run_single_two_state_search(eq_constraints: Callable,
                                ineq_constraints: Callable,
                                num_eps: int,
                                make_reward_fun: Callable,
                                policy_permutation: tuple[Policy],
                                env: MDPEnv):
    res = minimize(
        fun=lambda x: 0,
        x0=np.ones(REWARD_SIZE + num_eps),
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
        for i in range(REWARD_SIZE):
            print(f"{i}: {round(res.x[i], 2)}")
        print()
        print()


def search_make_reward_fun(dec_vars):
    rewards = dec_vars[:4].reshape((2, 2))

    def reward_fun(state, action):
        return rewards[state, action]

    return reward_fun


# for worse_policy, better_policy in policies_to_equate:
#     wp = make_two_state_policy(worse_policy)
#     bp = make_two_state_policy(better_policy)
#     run_search(equal_policy_pairs=[(wp, bp)],
#                make_reward_fun=search_make_reward_fun,
#                env=env)

# for p1, p2, p3 in triple_policies_to_equate:
#     pp1 = make_two_state_policy(p1)
#     pp2 = make_two_state_policy(p2)
#     pp3 = make_two_state_policy(p3)
#     run_search(equal_policy_pairs=[(pp1, pp2), (pp1, pp3)],
#                make_reward_fun=search_make_reward_fun,
#                env=env)

discount = 0.5
env = MDPEnv(dynamics=dynamics, discount=discount)

# Choose the set of policies and rewards to consider
policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
policy_funs = []
for policy in policies:
    policy_funs.append(make_two_state_policy(policy))

p00 = make_two_state_policy((0, 0))
p01 = make_two_state_policy((0, 1))
p10 = make_two_state_policy((1, 0))
p11 = make_two_state_policy((1, 1))

policy_permutation = [p00, p01, p10, p11]

rewards = (0, 2, 2, 0)


def temp_reward_fun(state, action):
    if (state, action) == (0, 0):
        return rewards[0]
    elif (state, action) == (0, 1):
        return rewards[1]
    elif (state, action) == (1, 0):
        return rewards[2]
    elif (state, action) == (1, 1):
        return rewards[3]
    else:
        raise SystemExit


# wp = make_two_state_policy((0, 0))
# bp = make_two_state_policy((1, 1))
# run_search(adjacent_policy_relations=[1, 0, 1],
#            equal_policy_pairs=[(wp, bp)],
#            make_reward_fun=search_make_reward_fun,
#            env=env)

achievable_permutations = calculate_achievable_permutations(allowed_policies=policy_permutation,
                                                            make_reward_fun=make_reward_fun_from_dec_vars,
                                                            env=env,
                                                            reward_size=REWARD_SIZE,
                                                            search_steps=SEARCH_STEPS,
                                                            show_rewards=True,
                                                            print_output=True)

adjacent_policy_relations = [2, 2, 2]

policies_to_equate = [(p00, p01)]

run_simplification_search(adjacent_policy_relations=adjacent_policy_relations,
                          equal_policy_list=policies_to_equate,
                          policy_permutations=achievable_permutations,
                          make_reward_fun=make_reward_fun_from_dec_vars,
                          run_single_env_specific_search=run_single_two_state_search,
                          env=env)

# all_ave_policy_vals = env.get_all_average_policy_values(policy_permutation=policy_permutation,
#                                                         reward_fun=temp_reward_fun)
# print(all_ave_policy_vals)
