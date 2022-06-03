# 2022 (c) Nikolaus Howe
import itertools
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from typing import Any, Callable

from environment import MDPWithoutRewardEnv
from policy import Policy
from utils import fancy_str_permutation


def check_gameable(policy_values_1, policy_values_2):
    for i, val in enumerate(policy_values_1):
        for j, val2 in enumerate(policy_values_1):
            if i == j:
                continue
            if (val < val2 and policy_values_2[i] > policy_values_2[j]) \
                    or (val > val2 and policy_values_2[i] < policy_values_2[j]):
                # print(f"{first} and {second} are gameable due to {translate[i]} and {translate[j]}")
                return True
    return False


def get_policy_values(policies: list[Policy], reward_fun: Callable, env: MDPWithoutRewardEnv):
    policies_and_values = []
    for policy in policies:
        value = env.get_average_policy_value(policy, reward_fun)
        policies_and_values.append((policy, value))
    return policies_and_values


def get_ungameable_policies_and_values(policies: list[Policy], reward_fun: Callable, env: MDPWithoutRewardEnv):
    policies_and_values = get_policy_values(policies, reward_fun, env)
    gameable_policies = []
    for i, first in enumerate(policies_and_values):
        for j, second in enumerate(policies_and_values[i + 1:]):
            if not check_gameable(first[1], second[1]):
                gameable_policies.append((first[0], second[0]))
    return gameable_policies


def make_ungameability_graph(ungameable_policy_pairs: list[tuple[Any, Any]]):
    edges = set()
    nodes = set()
    for first, second in ungameable_policy_pairs:
        edges.add((first, second))
        nodes.add(first)
        nodes.add(second)

    nodes = list(nodes)

    labels = {}
    for node in nodes:
        if type(node[0][0]) == Policy:
            labels[node] = fancy_str_permutation(*node)
        else:
            labels[node] = node

    plot_graph(nodes, edges, labels, title="Ungameability Graph")


def plot_graph(nodes, edges, labels, title="Graph"):
    print("the labels are", labels)

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    G = nx.Graph()
    G.add_edges_from(edges)

    options = {"edgecolors": "tab:blue", "node_size": 1500, "alpha": 1}

    pos = nx.spring_layout(G, seed=3113794651)  # positions for all nodes

    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:blue", **options)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=1,
        alpha=1,
        edge_color="tab:blue",
        node_size=1500,
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=4, font_color="black", font_weight="bold")

    plt.tight_layout()
    plt.axis("off")

    plt.savefig(f'{title}.pdf')


def remove_equivalent_orderings(orderings_and_relations: set[tuple[tuple[Policy], tuple[int]]]):
    orderings_and_relations = set(orderings_and_relations)

    to_remove = set()
    for i, ordering_and_relation_1 in enumerate(orderings_and_relations):
        for j, ordering_and_relation_2 in enumerate(orderings_and_relations):
            if i == j:
                continue
            else:
                if check_equivalent(*ordering_and_relation_1, *ordering_and_relation_2):
                    ordering1, relation1 = ordering_and_relation_1
                    ordering2, relation2 = ordering_and_relation_2
                    if ordering1 > ordering2:
                        to_remove.add((ordering1, relation1))
                    else:
                        to_remove.add((ordering2, relation2))

    print("all")

    return orderings_and_relations - to_remove  # set subtraction


def get_set_representation(ordering, relation):
    list_of_sets = [{ordering[0]}]
    for i, policy in enumerate(ordering[1:]):
        if relation[i] == 0:
            list_of_sets[-1].add(policy)
        else:
            assert relation[i] == 1
            list_of_sets.append({policy})

    return list_of_sets

def get_policy_set_index(policy: Policy, list_of_sets: list[set[Policy]]):
    for i, set_of_policies in enumerate(list_of_sets):
        if policy in set_of_policies:
            return i
    raise ValueError("Policy not found in list of sets")

def check_ungameable(ordering_and_relation_1, ordering_and_relation_2):
    list_of_sets_1 = get_set_representation(*ordering_and_relation_1)
    list_of_sets_2 = get_set_representation(*ordering_and_relation_2)

    for i, policy1 in enumerate(ordering_and_relation_1[0]):
        for j, policy2 in enumerate(ordering_and_relation_1[0]):
            if i == j:
                continue

            idx_p1_list1 = get_policy_set_index(policy1, list_of_sets_1)
            idx_p1_list2 = get_policy_set_index(policy1, list_of_sets_2)
            idx_p2_list1 = get_policy_set_index(policy2, list_of_sets_1)
            idx_p2_list2 = get_policy_set_index(policy2, list_of_sets_2)

            if idx_p1_list1 < idx_p2_list1 and idx_p1_list2 > idx_p2_list2:
                return False
    return True


def check_equivalent(ordering1: tuple[Policy], relation1: tuple[int],
                     ordering2: tuple[Policy], relation2: tuple[int]):
    # Immediately discard if the relations are different
    if relation1 != relation2:
        return False

    list_of_sets1 = get_set_representation(ordering1, relation1)
    list_of_sets2 = get_set_representation(ordering2, relation2)
    for i, set1 in enumerate(list_of_sets1):
        set2 = list_of_sets2[i]
        if set1 != set2:
            return False

    return True