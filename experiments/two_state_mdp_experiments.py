# (c) 2022 Nikolaus Howe
from mdp_env import MDPEnv
from permutations import calculate_achievable_permutations
from policy import make_two_state_policy
from simplification import run_full_simplification_search

REWARD_SIZE = 4  # four (s, a) pairs, different reward for each
REWARD_SHAPE = (2, 2)
SEARCH_STEPS = 1000


#################
# Two-state MDP #
#################

# It's a two-state two-action environment, with deterministic dynamics
def dynamics(state, action):
    del state
    return action


# Reward is deterministic and depends on state and action
def make_reward_fun_from_dec_vars(dec_vars):
    # Reward values are first in the decision variable array
    rewards = dec_vars[:REWARD_SIZE].reshape(REWARD_SHAPE)

    def reward_fun(state, action):
        return rewards[state, action]

    return reward_fun


def main():
    # Set up the MDP\R
    discount = 0.5
    env = MDPEnv(dynamics=dynamics, discount=discount)

    # Choose the set of policies and rewards to consider
    policies = [(0, 0), (0, 1), (1, 0), (1, 1)]

    policy_funs = []
    for policy in policies:
        policy_funs.append(make_two_state_policy(policy))
    p00, p01, p10, p11 = policy_funs

    # Specify which policies we want to choose among
    allowed_policies = policy_funs

    achievable_permutations = calculate_achievable_permutations(allowed_policies=allowed_policies,
                                                                make_reward_fun=make_reward_fun_from_dec_vars,
                                                                env=env,
                                                                reward_size=REWARD_SIZE,
                                                                search_steps=SEARCH_STEPS,
                                                                show_rewards=True,
                                                                print_output=True)

    # Enforce adjacent policy relations as desired
    # 0: =, 1: <, 2: not specified
    adjacent_policy_relations = [2, 2, 2]

    policies_to_equate = [(p00, p11)]

    run_full_simplification_search(adjacent_policy_relations=adjacent_policy_relations,
                                   equal_policy_list=policies_to_equate,
                                   policy_permutations=achievable_permutations,
                                   make_reward_fun=make_reward_fun_from_dec_vars,
                                   reward_size=REWARD_SIZE,
                                   env=env)


if __name__ == "__main__":
    main()
