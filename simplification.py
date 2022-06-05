# 2022 (c) Nikolaus Howe
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from typing import Any, Callable

from environment import MDPWithoutRewardEnv
from policy import Policy
from utils import fancy_str_permutation


def get_set_representation(ordering, relation):
    list_of_sets = [{ordering[0]}]
    for i, policy in enumerate(ordering[1:]):
        if relation[i] == 0:
            list_of_sets[-1].add(policy)
        else:
            assert relation[i] == 1
            list_of_sets.append({policy})

    return list_of_sets


def get_set_index(policy, list_of_sets):
    for i, s in enumerate(list_of_sets):
        if policy in s:
            return i
    raise ValueError("Policy not in list of sets")


# Check whether ordering_and_relation_1 is a simplification of ordering_and_relation_2
def check_simplification(ordering_and_relation_1, ordering_and_relation_2):
    list_of_sets_1 = get_set_representation(*ordering_and_relation_1)
    list_of_sets_2 = get_set_representation(*ordering_and_relation_2)

    policies = ordering_and_relation_1[0]

    found_different = False
    for policy1 in policies:
        for policy2 in policies:
            if policy1 == policy2:
                continue
            else:
                idx1 = get_set_index(policy1, list_of_sets_1)
                idx2 = get_set_index(policy2, list_of_sets_1)
                iidx1 = get_set_index(policy1, list_of_sets_2)
                iidx2 = get_set_index(policy2, list_of_sets_2)
                if idx1 < idx2 and iidx1 > iidx2:
                    return False
                elif idx1 == idx2 and iidx1 != iidx2:
                    return False
                if iidx1 == iidx2 and idx1 != idx2:
                    found_different = True

    return found_different


def get_policy_values(policies: list[Policy], reward_fun: Callable, env: MDPWithoutRewardEnv):
    policies_and_values = []
    for policy in policies:
        value = env.get_average_policy_value(policy, reward_fun)
        policies_and_values.append((policy, value))
    return policies_and_values


def get_simplifications_policies_and_values(policies: list[Policy], reward_fun: Callable, env: MDPWithoutRewardEnv):
    policies_and_values = get_policy_values(policies, reward_fun, env)
    simplified_policies = []
    for i, first in enumerate(policies_and_values):
        for j, second in enumerate(policies_and_values[i + 1:]):
            if not check_simplification(first[1], second[1]):
                simplified_policies.append((first[0], second[0]))
    return simplified_policies


def plot_graph(nodes, edges, labels, title="Graph"):
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    G = nx.DiGraph()
    G.add_edges_from(edges)

    options = {"edgecolors": "tab:blue", "node_size": 1000, "alpha": 1}

    pos = nx.spring_layout(G, seed=3113794651)  # positions for all nodes

    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:blue", **options)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=1,
        alpha=1,
        edge_color="tab:blue",
        node_size=1000,
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=4, font_color="black", font_weight="bold")

    plt.tight_layout()
    plt.axis("off")

    plt.savefig(f'{title}.pdf')
    plt.close()


# TODO: make graph layout work better
def make_simplification_graph(simplified_policy_pairs: list[tuple[Any, Any]]):
    edges = set()
    nodes = set()
    print("The simplifications are")
    for first, second in simplified_policy_pairs:
        edges.add((first, second))
        nodes.add(first)
        nodes.add(second)
        print(first, "->", second)

    nodes = list(nodes)

    labels = {}
    for node in nodes:
        if type(node[0][0]) == Policy:
            labels[node] = fancy_str_permutation(*node)
        else:
            labels[node] = node

    plot_graph(nodes, edges, labels, title="Simplification Graph")
