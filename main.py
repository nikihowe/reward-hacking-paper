# (c) 2022 Nikolaus Howe
import itertools
import numpy as np

from scipy.optimize import minimize

from mdp_env import MDPEnv
from policy import Policy, make_cleaning_policy, make_two_state_policy
import utils


# Set up the MDP Enviroment
def dynamics(state, action):
    del state
    return action


discount = 0.5
env = MDPEnv(dynamics=dynamics, discount=discount)

# Choose the set of policies and rewards to consider
policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
policy_funs = []
for policy in policies:
    policy_funs.append(make_two_state_policy(policy))


# for policy_fun in policy_funs:
#     print("looking at fun", policy_fun)
#     print("input 0, output", policy_fun(0))
#     print("input 1, output", policy_fun(1))
#     print()

# simple_rewards = np.array([[0, 1],
#                            [2, 3]])
# reward_fun = lambda s, a: simple_rewards[s, a]
#
# print(env.get_all_average_policy_values(policy_permutation=policy_funs, reward_fun=reward_fun))
# print(env.get_sorted_policies_and_rewards(policies=policy_funs, reward_fun=reward_fun))
# print(env.get_ineqs_from_policies_and_rewards(policies=policy_funs, reward_fun=reward_fun))
# raise SystemExit


def run_search(worse_policy: Policy, better_policy: Policy, make_reward_fun, env):
    print("policy funs being equated:", worse_policy, better_policy)

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

        eq_constraints = utils.make_specific_eq_constraints(env=env,
                                                            make_reward_fun=make_reward_fun,
                                                            equal_policy_pairs=[(worse_policy, better_policy)]
                                                            )
        ineq_constraints = utils.make_ineq_constraints(policy_permutation=policy_permutation,
                                                       make_reward_fun=make_reward_fun,
                                                       num_eps=4,
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

        print("#######################################################")

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
#     run_search(worse_policy=Policy(worse_policy, lambda s: worse_policy[s]),
#                better_policy=Policy(better_policy, lambda s: better_policy[s]),
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


# r00 = 1
# r01 = 0
# r10 = 0
# r11 = 1
#
# rrr = np.array([[r00, r01],
#                 [r10, r11]])
#
# def rrr_fun(state, action):
#     return rrr[state, action]
#
# print(env.get_all_average_policy_values(policy_funs, rrr_fun))


##################
# Cleaning robot #
##################

# It's a one-step bandit, so we're going to pretend there is only one state: 0
def cleaning_dynamics(state, action):
    del state, action
    return 0


def make_cleaning_reward_fun(rewards):
    def reward_fun(state, action):
        del state
        return action @ rewards

    return reward_fun


def search_make_cleaning_reward_fun(dec_vars):
    rewards = dec_vars[:3]

    def reward_fun(state, action):
        del state
        return action @ rewards

    return reward_fun


def run_cleaning_search(equal_policy_list: list[tuple[Policy, Policy]],
                        allowed_policies: list[Policy],
                        make_reward_fun, env) -> None:
    print("policy funs being equated:", equal_policy_list)

    double_permutations = list(itertools.permutations(allowed_policies))
    all_permutations = set()
    lower_policy = make_cleaning_policy((0, 0, 0))
    higher_policy = make_cleaning_policy((1, 1, 1))

    # print("checking cleaning policy outputs")
    # print(lower_policy(0))
    # print(lower_policy(1))
    # print(higher_policy(0))
    for perm in double_permutations:
        if perm.index(lower_policy) < perm.index(higher_policy):
            all_permutations.add(perm)

    # print("all permutations", all_permutations)
    considered_permutations = set()

    for _ in range(1000):
        # Generate random decision variables
        # reward: 3, epsilons: 7
        dec_vars = np.random.uniform(0, 1, 10)  # rewards and epsilon
        # dec_vars[:3] = [1, 2, 3]
        # print("rewards are", dec_vars[:3])

        # Check whether we've already seen this permutation
        temp_reward_fun = make_reward_fun(dec_vars[:3])
        policies_and_rewards = env.get_sorted_policies_and_rewards(allowed_policies, reward_fun=temp_reward_fun)
        # print("policies and rewards:")
        # for par in policies_and_rewards:
        #     print(par)

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
                                                            equal_policy_pairs=equal_policy_list)
        ineq_constraints = utils.make_ineq_constraints(policy_permutation=policy_permutation,
                                                       make_reward_fun=make_reward_fun,
                                                       num_eps=7,
                                                       env=env)  # pp or policies?

        res = minimize(
            fun=lambda vars_and_epsilons: -np.sum(vars_and_epsilons[3:]),  # sum of epsilons
            x0=np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
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

        print("#######################################################")

    print("possible permutations")
    considered_permutations = sorted(list(considered_permutations))
    print("num:", len(considered_permutations))
    for perm in considered_permutations:
        print(perm)

    print()

    # print("impossible permutations")
    # all_permutations = sorted(list(all_permutations))
    # print("num:", len(all_permutations))
    # for perm in all_permutations:
    #     print(perm)


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

# for cp in cleaning_policy_funs:
#     print("pp", cp)

cleaning_env = MDPEnv(dynamics=cleaning_dynamics, discount=0)

p000 = make_cleaning_policy((0, 0, 0))
p001 = make_cleaning_policy((0, 0, 1))
p100 = make_cleaning_policy((1, 0, 0))
p110 = make_cleaning_policy((1, 1, 0))
p111 = make_cleaning_policy((1, 1, 1))

new_allowed_policies = [p000, p001, p100, p110, p111]

# policies_to_equate = [(p000, p001), (p100, p110)]
policies_to_equate = [(p000, p110)]

run_cleaning_search(equal_policy_list=policies_to_equate,
                    allowed_policies=new_allowed_policies,
                    make_reward_fun=search_make_cleaning_reward_fun,
                    env=cleaning_env)
