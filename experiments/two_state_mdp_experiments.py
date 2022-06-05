# (c) 2022 Nikolaus Howe
from environment import MDPWithoutRewardEnv
from gameability import make_ungameability_graph, check_ungameable, remove_equivalent_orderings
from permutations import calculate_achievable_permutations
from policy import Policy, make_two_state_policy
from policy_ordering import run_adjacent_relation_search, run_full_ordering_search
from simplification import check_simplification, make_simplification_graph

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
def make_reward_fun_from_dec_vars(reward_components):
    # Reward values are first in the decision variable array
    rewards = reward_components.reshape(REWARD_SHAPE)

    def reward_fun(state, action):
        return rewards[state, action]

    return reward_fun


def fancy_print(perm, relation, reward):
    toprint = []
    for i, p in enumerate(perm):
        if str(p) == "Policy(0, 0)":
            toprint.append("p00")
        elif str(p) == "Policy(0, 1)":
            toprint.append("p01")
        elif str(p) == "Policy(1, 0)":
            toprint.append("p10")
        elif str(p) == "Policy(1, 1)":
            toprint.append("p11")

        if i < len(relation):
            toprint.append(" < " if relation[i] == 1 else " = ")

    print("".join(toprint), ":", reward)


def run_two_state_mdp_experiment():
    # Set up the MDP\R
    discount = 0.5
    env = MDPWithoutRewardEnv(dynamics=dynamics, discount=discount)

    # Choose the set of policies and rewards to consider
    policies = [(0, 0), (0, 1), (1, 0), (1, 1)]

    policy_funs = []
    for policy in policies:
        policy_funs.append(make_two_state_policy(policy))
    p00, p01, p10, p11 = policy_funs

    # Specify which policies we want to choose among
    allowed_policies = policy_funs

    realized_permutations, realized_relations, realized_rewards = calculate_achievable_permutations(
        allowed_policies=allowed_policies,
        make_reward_fun=make_reward_fun_from_dec_vars,
        env=env,
        reward_size=REWARD_SIZE, )

    # keep = []
    # for i, perm in enumerate(realized_permutations):
    #     if perm.index(p00) > perm.index(p11):
    #         continue
    #     if realized_relations[i] == (0, 0, 0):
    #         continue
    #     keep.append((perm, realized_relations[i], realized_rewards[i]))
    #     fancy_print(perm, realized_relations[i], realized_rewards[i])
    #
    # print("num perms", len(keep))

    successful_orderings_with_relations = run_full_ordering_search(policies=allowed_policies,
                                                                   make_reward_fun=make_reward_fun_from_dec_vars,
                                                                   reward_size=REWARD_SIZE,
                                                                   env=env)

    # Remove equivalent orderings
    orderings_and_relations = remove_equivalent_orderings(set(successful_orderings_with_relations))

    # Make ungameability graph
    ungameable_pairs = set()
    for i, ordering_and_relation1 in enumerate(orderings_and_relations):
        for j, ordering_and_relation2 in enumerate(orderings_and_relations):
            if i == j:
                continue

            if check_ungameable(ordering_and_relation1, ordering_and_relation2):
                ungameable_pairs.add((ordering_and_relation1, ordering_and_relation2))

    make_ungameability_graph(list(ungameable_pairs))

    # Make simplification graph
    simplification_pairs = set()
    for i, ordering_and_relation1 in enumerate(orderings_and_relations):
        for j, ordering_and_relation2 in enumerate(orderings_and_relations):
            if i == j:
                continue

            if check_simplification(ordering_and_relation1, ordering_and_relation2):
                simplification_pairs.add((ordering_and_relation1, ordering_and_relation2))

    make_simplification_graph(list(simplification_pairs))


if __name__ == "__main__":
    run_two_state_mdp_experiment()
