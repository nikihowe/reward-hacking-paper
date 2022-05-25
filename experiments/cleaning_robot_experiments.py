# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize
from typing import Callable

from mdp_env import MDPEnv
from policy import Policy, make_cleaning_policy, make_two_state_policy
import utils

SEARCH_STEPS = 200


##################
# Cleaning robot #
##################

# It's a one-step eight-arm bandit, so we're going to pretend there is only one state: 0
def cleaning_dynamics(state, action):
    del state, action
    return 0


# Reward is the sum of rooms that are cleaned
def make_cleaning_reward_fun(rewards):
    def reward_fun(state, action):
        del state
        return action @ rewards

    return reward_fun


# Make a reward function from decision variables
def search_make_cleaning_reward_fun(dec_vars):
    rewards = dec_vars[:3]

    def reward_fun(state, action):
        del state
        return action @ rewards

    return reward_fun


# Calculate which permutations are possible
def calculate_achievable_permutations(allowed_policies: list[Policy],
                                      make_reward_fun: Callable,
                                      env: MDPEnv,
                                      show_rewards: bool = False,
                                      print_output: bool = False) -> list[tuple[Policy]]:
    print("Considering all orderings of the following policies:")
    print(allowed_policies, '\n--------------------------------\n')

    num_eps = len(allowed_policies) - 1

    all_permutations = list(itertools.permutations(allowed_policies))

    considered_permutations = []

    for _ in range(SEARCH_STEPS):
        dec_vars = np.random.uniform(0, 1, 3 + num_eps)  # rewards and epsilons

        # Check whether we've already seen this permutation
        temp_reward_fun = make_reward_fun(dec_vars[:num_eps])
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


def run_single_cleaning_search(eq_constraints: Callable,
                               ineq_constraints: Callable,
                               num_eps: int,
                               make_reward_fun: Callable,
                               policy_permutation: list[Policy],
                               env: MDPEnv):
    res = minimize(
        fun=lambda vars_and_epsilons: np.minimum(-np.sum(vars_and_epsilons[-num_eps:]), -10),  # sum of epsilons
        # fun=lambda x: 0,
        x0=np.ones(3 + num_eps),
        constraints=
        [{"type": "eq",
          "fun": eq_constraints},
         {"type": "ineq",
          "fun": ineq_constraints}]
    )
    if res.success:
        print(res.x)
        print("the values of the policies are")
        temp_rf = make_reward_fun(res.x)
        all_ave_policy_vals = env.get_all_average_policy_values(policy_permutation=policy_permutation,
                                                                reward_fun=temp_rf)
        for i, policy in enumerate(policy_permutation):
            print(f"{policy}: {all_ave_policy_vals[i]}")
        print()


def run_cleaning_search(adjacent_policy_relations: list[int],
                        equal_policy_list: list[tuple[Policy, Policy]],
                        policy_permutations: list[tuple],
                        make_reward_fun: Callable,
                        env: MDPEnv) -> None:
    print("policy funs being equated:", equal_policy_list)
    num_eps = len(policy_permutations[0]) - 1
    print("num eps is", num_eps)

    for policy_permutation in policy_permutations:
        print(f"Given permutation: {policy_permutation}")
        print(f"Trying to set equal {equal_policy_list}")
        print(f"Adjacent policy relations: {adjacent_policy_relations}")

        eq_constraints = utils.make_specific_eq_constraints(env=env,
                                                            policy_permutation=policy_permutation,
                                                            make_reward_fun=make_reward_fun,
                                                            equal_policy_pairs=equal_policy_list,
                                                            num_eps=num_eps,
                                                            adjacent_policy_relations=adjacent_policy_relations)
        ineq_constraints = utils.make_ineq_constraints(adjacent_policy_relations=adjacent_policy_relations,
                                                       policy_permutation=policy_permutation,
                                                       make_reward_fun=make_reward_fun,
                                                       num_eps=num_eps,
                                                       env=env)
        run_single_cleaning_search(eq_constraints=eq_constraints,
                                   ineq_constraints=ineq_constraints,
                                   num_eps=num_eps,
                                   make_reward_fun=make_reward_fun,
                                   policy_permutation=policy_permutation,
                                   env=env)

        print("#######################################################")


cleaning_policies = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1)]

cleaning_policy_funs = []
for cleaning_policy_list in cleaning_policies:
    cleaning_policy_funs.append(make_cleaning_policy(cleaning_policy_list))

cleaning_env = MDPEnv(dynamics=cleaning_dynamics, discount=0)

p000 = make_cleaning_policy((0, 0, 0))
p001 = make_cleaning_policy((0, 0, 1))
p010 = make_cleaning_policy((0, 1, 0))
p011 = make_cleaning_policy((0, 1, 1))
p100 = make_cleaning_policy((1, 0, 0))
p101 = make_cleaning_policy((1, 0, 1))
p110 = make_cleaning_policy((1, 1, 0))
p111 = make_cleaning_policy((1, 1, 1))

new_allowed_policies = [p000, p001, p100, p110, p111]

# all_allowed_policies = [p000,
#                         p001,
#                         p010,
#                         p011,
#                         p100,
#                         p101,
#                         p110,
#                         p111]

# policies_to_equate = [(p000, p001), (p100, p110)]
policies_to_equate = [(p000, p100)]

# adjacent_policy_relations = [0, 0, 0, 1]
adjacent_policy_relations = [0, 0, 1, 0]

# run_cleaning_search(adjacent_policy_relations=adjacent_policy_relations,
#                     equal_policy_list=policies_to_equate,
#                     allowed_policies=all_allowed_policies,
#                     make_reward_fun=search_make_cleaning_reward_fun,
#                     env=cleaning_env)


# achievable_permutations = calculate_achievable_permutations(new_allowed_policies,
#                                                             make_reward_fun=search_make_cleaning_reward_fun,
#                                                             env=cleaning_env,
#                                                             show_rewards=True,
#                                                             print_output=True)

achievable_permutations = [(p000, p001, p100, p110, p111),
                           (p000, p100, p001, p110, p111),
                           (p000, p100, p110, p001, p111)]

run_cleaning_search(adjacent_policy_relations=adjacent_policy_relations,
                    equal_policy_list=policies_to_equate,
                    policy_permutations=achievable_permutations,
                    make_reward_fun=search_make_cleaning_reward_fun,
                    env=cleaning_env)
