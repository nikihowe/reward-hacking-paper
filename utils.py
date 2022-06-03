# (c) 2022 Nikolaus Howe
def fancy_print_permutation(policy_permutation, adjacent_policy_relations, realized_rewards=None):
    for i, policy in enumerate(policy_permutation[:-1]):
        relation = '=' if adjacent_policy_relations[i] == 0 else '<' if adjacent_policy_relations[i] == 1 else '?'
        print(policy, relation, end=' ')
    print(policy_permutation[-1], end=' ')
    if realized_rewards is not None:
        print(realized_rewards)
    else:
        print()


def extract_short_policy_name(policy):
    short = []
    policy_name = policy.get_name()
    for char in policy_name:
        short.append(str(char))
    return ''.join(short)


def fancy_str_permutation(policy_permutation, adjacent_policy_relations, realized_rewards=None):
    the_string = ['$']
    for i, policy in enumerate(policy_permutation[:-1]):
        short = extract_short_policy_name(policy)
        the_string.append(short)

        relation = '=' if adjacent_policy_relations[i] == 0 else '<' if adjacent_policy_relations[i] == 1 else '?'
        the_string.append(relation)
    the_string.append(extract_short_policy_name(policy_permutation[-1]))
    the_string.append('$')
    return ''.join(the_string)
