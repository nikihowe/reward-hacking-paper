# (c) 2022 Nikolaus Howe
from mdp_env import MDPEnv
from permutations import calculate_achievable_permutations
from policy import make_cleaning_policy
from simplification import run_full_ordering_search

REWARD_SIZE = 3  # three rooms, so three reward components
SEARCH_STEPS = 2000


##################
# Cleaning robot #
##################

# It's a one-step eight-arm bandit, so we're going to pretend there is only one state: 0
def cleaning_dynamics(state, action):
    del state, action
    return 0


# Make a reward function from decision variables
def make_reward_fun(rewards):
    def reward_fun(state, action):
        del state
        return action @ rewards

    return reward_fun


def run_cleaning_robot_experiment():
    # Set up the MDP\R
    cleaning_env = MDPEnv(dynamics=cleaning_dynamics, discount=0)

    # Choose the set of policies and rewards to consider
    policies = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)]

    policy_funs = []
    for policy_list in policies:
        policy_funs.append(make_cleaning_policy(policy_list))
    p000, p001, p010, p011, p100, p101, p110, p111 = policy_funs

    # Specify which policies we want to choose among
    # allowed_policies = policy_funs
    allowed_policies = [p000, p001, p110, p111]

    # try:
    #     with open('cleaning_robot_experiment.pkl', 'rb') as f:
    #         achievable_permutations = pkl.load(f)
    # except FileNotFoundError:
    # achievable_permutations = calculate_achievable_permutations(allowed_policies=allowed_policies,
    #                                                             make_reward_fun=make_reward_fun,
    #                                                             env=cleaning_env,
    #                                                             reward_size=REWARD_SIZE, )

    # pkl.dump(achievable_permutations, open('cleaning_robot_permutations.pkl', 'wb'))

    run_full_ordering_search(policies=allowed_policies,
                             make_reward_fun=make_reward_fun,
                             reward_size=REWARD_SIZE,
                             env=cleaning_env)


# achievable_permutations = [(p000, p001, p100, p110, p111),
#                            (p000, p100, p001, p110, p111),
#                            (p000, p100, p110, p001, p111)]

# Note that for the cleaning robot example, the mathematical program solver
# is not able to consistently find all the solutions. It would be interesting
# to better understand why this is the case (since in the two-state
# example, it finds them all without problem).

# Enforce adjacent policy relations as desired
# The adjacent_policy_relations list must be of length len(allowed_policies) - 1
# 0: =, 1: <, 2: <=
# adjacent_policy_relations = [2, 2, 2, 2, 2, 2, 2]
# adjacent_policy_relations = [1, 0, 1, 1]
#

if __name__ == '__main__':
    run_cleaning_robot_experiment()
