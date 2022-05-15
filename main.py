# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize

from mdp_env import MDPEnv
import utils

# Set up the MDP Enviroment
states = np.array([0, 1])
actions = np.array([0, 1])
discount = 0.5
env = MDPEnv(states=states, actions=actions, discount=discount)

# Choose the set of policies and rewards to consider
policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
simple_rewards = np.array([[0, 1],
                           [2, 3]])


# print(env.get_all_average_policy_values(policies, simple_rewards))
# print(env.get_sorted_policies_and_rewards(policies, simple_rewards))
# print(env.get_ineqs_from_policies_and_rewards(policies, simple_rewards))


def run_search(worse_policy, better_policy, make_reward_fun, env):
    print("policies being equated:", worse_policy, better_policy)

    double_all_permutations = list(itertools.permutations(policies))
    all_permutations = set()

    for i, perm in enumerate(double_all_permutations):
        if perm.index((0, 0)) < perm.index((1, 1)):
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
        policies_and_rewards = env.get_sorted_policies_and_rewards(policies, reward_fun=temp_reward_fun)
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

        eq_constraints = utils.make_specific_eq_constraints(env=env,
                                                            make_reward_fun=make_reward_fun,
                                                            worse_policy=worse_policy,
                                                            better_policy=better_policy)
        ineq_constraints = utils.make_ineq_constraints(policy_permutation=policy_permutation,
                                                       make_reward_fun=make_reward_fun,
                                                       env=env)  # pp or policies?

        res = minimize(
            fun=lambda vars_and_epsilons: (- vars_and_epsilons[-3]
                                           - vars_and_epsilons[-2]
                                           - vars_and_epsilons[-1]),  # sum of epsilons
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
            print(res.x)
            print("the values of the policies are")
            temp_rf = make_reward_fun(res.x)
            all_ave_policy_vals = env.get_all_average_policy_values(policy_permutation=policy_permutation,
                                                                    reward_fun=temp_rf)
            for i, policy in enumerate(policy_permutation):
                print(f"{policy}: {all_ave_policy_vals[i]}")
            print()

        # print("#######################################################")

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

# policies_to_equate = [((0, 0), (0, 1)),
#                       ((0, 0), (1, 0)),
#                       ((0, 0), (1, 1)),
#                       ((0, 1), (1, 0)),
#                       ((0, 1), (1, 1)),
#                       ((1, 0), (1, 1))]
#
#
# def search_make_reward_fun(dec_vars):
#     rewards = dec_vars[:4].reshape((2, 2))
#
#     def reward_fun(state, action):
#         return rewards[state, action]
#
#     return reward_fun
#
#
# for worse_policy, better_policy in policies_to_equate:
#     run_search(worse_policy=worse_policy,
#                better_policy=better_policy,
#                make_reward_fun=search_make_reward_fun,
#                env=env)

###############################

# guess_dec_vars = np.array([7.14, 3.39, 21.5, 15.5, 1.02, 2.13, 0.])
#
# eq_constraints = utils.make_specific_eq_constraints(env=env,
#                                                     worse_policy=(1, 0),
#                                                     better_policy=(1, 1))
# ineq_constraints = utils.make_ineq_constraints(policy_permutation=policies, env=env)
#
# res = minimize(
#     fun=lambda vars_and_epsilons: (- vars_and_epsilons[-3]
#                                    - vars_and_epsilons[-2]
#                                    - vars_and_epsilons[-1]),  # sum of epsilons
#     x0=guess_dec_vars,
#     constraints=
#     [{"type": "eq",
#       "fun": eq_constraints},
#      {"type": "ineq",
#       "fun": ineq_constraints}]
#     #     {"type": "ineq",
#     #       "fun": ineq_constraints}
# )
#
# final_dec_vars = res.x
#
# temp_rewards = final_dec_vars[:4].reshape(2, 2)
# print("final rewards")
# print(temp_rewards)
# print("policy values")
# pol_vals = env.get_all_average_policy_values(policies, temp_rewards)
# for i, policy in enumerate(policies):
#     print(f"{policy}: {pol_vals[i]}")

# print("#######################################################")

# guess_dec_vars = np.array([17.65, 7.75, 22.60, 5.18, 0., 2.13, 1.02])

# eq_constraints = utils.make_specific_eq_constraints(env=env,
#                                                     worse_policy=(0, 0),
#                                                     better_policy=(0, 1))
# ineq_constraints = utils.make_ineq_constraints(policies=policies, env=env)

# res = minimize(
#     fun=lambda vars_and_epsilons: (- vars_and_epsilons[-3]
#                                    - vars_and_epsilons[-2]
#                                    - vars_and_epsilons[-1]),  # sum of epsilons
#     x0=guess_dec_vars,
#     constraints=
#     [{"type": "eq",
#       "fun": eq_constraints},
#      {"type": "ineq",
#       "fun": ineq_constraints}]
#     #     {"type": "ineq",
#     #       "fun": ineq_constraints}
# )

# final_dec_vars = res.x

# temp_rewards = np.array([[17.65402037, 22.6045029],
#                          [5.18305534, 7.75305534]])
#
# flat_rewards = temp_rewards.flatten()
# print(flat_rewards)

# for perm in itertools.permutations(range(4)):
#     print(*perm)
#     r = flat_rewards[list(perm)].reshape(2, 2)
#     print("final rewards")
#     print(temp_rewards)
#     print("policy values")
#     print("policies", policies)
#     pol_vals = env.get_all_average_policy_values(policies, r)
#     for i, policy in enumerate(policies):
#         print(f"{policy}: {pol_vals[i]}")



r00 = 1
r01 = 0
r10 = 0
r11 = 1

rrr = np.array([[r00, r01],
                [r10, r11]])

def rrr_fun(state, action):
    return rrr[state, action]

print(env.get_all_average_policy_values(policies, rrr_fun))
