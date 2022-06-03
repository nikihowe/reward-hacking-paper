# (c) 2022 Nikolaus Howe
from environment import MDPWithoutRewardEnv
from gameability import remove_equivalent_orderings, check_ungameable, make_ungameability_graph
from permutations import calculate_achievable_permutations
from policy import make_cleaning_policy
from policy_ordering import run_full_ordering_search

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
    cleaning_env = MDPWithoutRewardEnv(dynamics=cleaning_dynamics, discount=0, num_states=1, num_actions=8,
                                       require_nonnegative_reward=True)

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
    allowed_policies = [p001, p110, p111]

    achievable_permutations = calculate_achievable_permutations(allowed_policies=allowed_policies,
                                                                make_reward_fun=make_reward_fun,
                                                                env=cleaning_env,
                                                                reward_size=REWARD_SIZE, )

    successful_orderings_with_relations = run_full_ordering_search(policies=allowed_policies,
                                                                   make_reward_fun=make_reward_fun,
                                                                   reward_size=REWARD_SIZE,
                                                                   env=cleaning_env)

    # Remove equivalent orderings
    orderings_and_relations = remove_equivalent_orderings(set(successful_orderings_with_relations))

    # Get ungameable pairs
    ungameable_pairs = set()
    for i, ordering_and_relation1 in enumerate(orderings_and_relations):
        for j, ordering_and_relation2 in enumerate(orderings_and_relations):
            if i == j:
                continue

            if check_ungameable(ordering_and_relation1, ordering_and_relation2):
                ungameable_pairs.add((ordering_and_relation1, ordering_and_relation2))

    print("ungameable", ungameable_pairs)

    make_ungameability_graph(list(ungameable_pairs))


if __name__ == '__main__':
    run_cleaning_robot_experiment()
