# 2022 (c) Nikolaus Howe
import itertools
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx



translate = {
    0: 'a',
    1: 'b',
    2: 'c',
}


def check_gameable(first, second):
    for i, el in enumerate(first):
        for j, el2 in enumerate(first):
            if i == j:
                continue
            if (el < el2 and second[i] > second[j]) or (el > el2 and second[i] < second[j]):
                # print(f"{first} and {second} are gameable due to {translate[i]} and {translate[j]}")
                return True
    return False


def values_to_string(ordering_and_relation):
    ordering, relation = ordering_and_relation
    string = []
    string.append("$")
    string.append(ordering[0])
    string.append(' = ' if relation[0] == 0 else ' < ')
    string.append(ordering[1])
    string.append(' = ' if relation[1] == 0 else ' < ')
    string.append(ordering[2])
    string.append("$")
    return ''.join(string)


orderings = list(itertools.permutations(['a', 'b', 'c']))
relations = list(itertools.product(*([range(2)] * 2)))
weak_orderings = []
for perm in orderings:
    for relation in relations:
        weak_orderings.append((perm, relation))

value_assignments = set()
corresponding_ordering_and_relation = {}
for ordering, relation in weak_orderings:
    value_assignment = {}

    cur_val = 0
    value_assignment[ordering[0]] = cur_val
    cur_val += relation[0]
    value_assignment[ordering[1]] = cur_val
    cur_val += relation[1]
    value_assignment[ordering[2]] = cur_val
    result = (value_assignment['a'], value_assignment['b'], value_assignment['c'])
    if result not in corresponding_ordering_and_relation:
        corresponding_ordering_and_relation[result] = (ordering, relation)
    value_assignments.add(result)

# for line in value_assignments:
#     print(line, corresponding_ordering_and_relation[line], values_to_string(corresponding_ordering_and_relation[line]))
value_assignment_pairs = []
for first in value_assignments:
    for second in value_assignments:
        if first == second:
            continue
        value_assignment_pairs.append((first, second))

edges = set()
nodes = set()
for first, second in value_assignment_pairs:
    first_string = values_to_string(corresponding_ordering_and_relation[first])
    second_string = values_to_string(corresponding_ordering_and_relation[second])
    if not check_gameable(first, second):
        edges.add((first_string, second_string))
        nodes.add(first_string)
        nodes.add(second_string)

nodes = list(nodes)

labels = {}
for node in nodes:
    labels[node] = node

for edge in edges:
    print(edge)


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

G = nx.Graph()
G.add_edges_from(edges)

# A = nx.nx_agraph.to_agraph(G)
# A.draw(G, prog="neato")
options = {"edgecolors": "tab:blue", "node_size": 1300, "alpha": 1}

pos = nx.spring_layout(G, seed=3113794651)  # positions for all nodes
# pos = nx.planar_layout(G)  # positions for all nodes

pos['$a < b < c$'] = (1.3, 0)
pos['$a = b < c$'] = (1.8, 0.8)
pos['$b < a < c$'] = (0.5, 0.7)
pos['$b < a = c$'] = (0, 1.5)
pos['$b < c < a$'] = (-0.5, 0.7)
pos['$b = c < a$'] = (-1.8, 0.8)
pos['$c < b < a$'] = (-1.3, 0)
pos['$c < a = b$'] = (-1.8, -0.8)
pos['$c < a < b$'] = (-0.5, -0.7)
pos['$a = c < b$'] = (0, -1.5)
pos['$a < c < b$'] = (0.5, -0.7)
pos['$a < b = c$'] = (1.8, -0.8)

pos['$a = b = c$'] = (0, 0)

nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:blue", **options)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=edges,
    width=1,
    alpha=1,
    edge_color="tab:blue",
    node_size=1300,
)
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white", font_weight="bold")

plt.tight_layout()
plt.axis("off")

plt.savefig('simple_ungameability.pdf')




