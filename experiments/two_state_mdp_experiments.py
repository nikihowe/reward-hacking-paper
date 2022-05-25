# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from mdp_env import MDPEnv
from permutations import calculate_achievable_permutations
from policy import Policy, make_two_state_policy
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
    print("X0 is of shape", REWARD_SIZE + num_eps)
    res = minimize(
        fun=lambda x: 0,
        # x0=np.array([.1, .1, .1, .1, .1, .1, .1]),
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


def run_search(adjacent_policy_relations: list[int],
               equal_policy_pairs: list[tuple[Policy, Policy]],
               make_reward_fun: Callable,
               env: MDPEnv,
               num_eps=3):
    print("policy funs being equated:", equal_policy_pairs)

    double_all_permutations = list(itertools.permutations(policy_funs))
    all_permutations = set()

    for i, perm in enumerate(double_all_permutations):
        if perm.index(make_two_state_policy((0, 0))) < perm.index(make_two_state_policy((1, 1))):
            all_permutations.add(perm)

    considered_permutations = set()

    for _ in range(1000):
        # if _ % 100 == 0:
        #     print(f"\n ~~~iter {_}~~~ \n")

        # Generate random decision variables
        dec_vars = np.random.uniform(0, 1, 7)  # rewards and epsilon

        # Check whether we've already seen this permutation
        # rewards = dec_vars[:4].reshape(2, 2)
        temp_reward_fun = make_reward_fun(dec_vars)
        policies_and_rewards = env.get_sorted_policies_and_rewards(policy_funs, reward_fun=temp_reward_fun)
        # print("policies and rewards:", policies_and_rewards)

        policy_permutation = []
        for (p, r) in policies_and_rewards:
            policy_permutation.append(p)
        policy_permutation = tuple(policy_permutation)
        if policy_permutation in all_permutations:
            print(f"running {_}: {policy_permutation}; {len(all_permutations)} left")
            all_permutations.remove(policy_permutation)
            # print("considered permutations:", considered_permutations)
            considered_permutations.add(policy_permutation)
        else:
            continue

        # adjacent_policy_relations = [0, 0, 1]  # 0 equality, 1 inequality, 2 unspecified
        policy_permutation = list(policy_permutation)
        print("policy permutation:", policy_permutation)

        eq_constraints = utils.make_specific_eq_constraints(env=env,
                                                            policy_permutation=policy_permutation,
                                                            make_reward_fun=make_reward_fun,
                                                            equal_policy_pairs=equal_policy_pairs,
                                                            num_eps=num_eps,
                                                            adjacent_policy_relations=adjacent_policy_relations)
        ineq_constraints = utils.make_ineq_constraints(policy_permutation=policy_permutation,
                                                       make_reward_fun=make_reward_fun,
                                                       num_eps=num_eps,
                                                       adjacent_policy_relations=adjacent_policy_relations,
                                                       env=env)  # pp or policies?

        res = minimize(
            # fun=lambda vars_and_epsilons: np.minimum((- vars_and_epsilons[-3]
            #                                           - vars_and_epsilons[-2]
            #                                           - vars_and_epsilons[-1]), -10),  # sum of epsilons
            fun=lambda x: 0,
            x0=np.array([.1, .1, .1, .1, .1, .1, .1]),
            constraints=
            [{"type": "eq",
              "fun": eq_constraints},
             {"type": "ineq",
              "fun": ineq_constraints}]
            #     {"type": "ineq",
            #       "fun": ineq_constraints}
        )
        if res.success:
            for i, num in enumerate(res.x):
                if i < len(dec_vars) - num_eps:
                    print(f"r_{i}: {round(num, 2)}")
                else:
                    print(f"eps_{i}: {round(num, 2)}")
            # print(res.x)
            print("the values of the policies are")
            temp_rf = make_reward_fun(res.x)
            all_ave_policy_vals = env.get_all_average_policy_values(policy_permutation=policy_permutation,
                                                                    reward_fun=temp_rf)
            for i, policy in enumerate(policy_permutation):
                print(f"{policy}: {round(all_ave_policy_vals[i], 2)}")
            print()

        print("-----------")
    print("###########################################")

    # print("possible permutations")
    # considered_permutations = sorted(list(considered_permutations))
    # for perm in considered_permutations:
    #     print(perm)
    #
    # print()
    #
    # print("impossible permutations")
    # all_permutations = sorted(list(all_permutations))
    # for perm in all_permutations:
    #     print(perm)


#####################
#####################
#####################

policies_to_equate = [((0, 0), (0, 1)),
                      ((0, 0), (1, 0)),
                      ((0, 0), (1, 1)),
                      ((0, 1), (1, 0)),
                      ((0, 1), (1, 1)),
                      ((1, 0), (1, 1))]


# triple_policies_to_equate = [((0, 0), (0, 1), (1, 0)),
#                              ((0, 0), (0, 1), (1, 1)),
#                              ((0, 0), (1, 0), (1, 1)),
#                              ((0, 1), (1, 0), (1, 1))]


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

out = calculate_achievable_permutations(allowed_policies=policy_permutation,
                                        make_reward_fun=make_reward_fun_from_dec_vars,
                                        env=env,
                                        reward_size=REWARD_SIZE,
                                        search_steps=SEARCH_STEPS,
                                        show_rewards=True,
                                        print_output=True)

# all_ave_policy_vals = env.get_all_average_policy_values(policy_permutation=policy_permutation,
#                                                         reward_fun=temp_reward_fun)
# print(all_ave_policy_vals)
