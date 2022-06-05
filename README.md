# Calculating Reward Function Permutations and Simplifications

This repository currently supports two tasks.

1) Given a set of policies, it calculates which of the permutations of these policies could actually arise as an ordering due to a reward function.
2) Given an ordering of policies and a set of policies to set equal, it attempts to find a reward function which satisfies the ordering while setting equal the required policies.

### Installation

```bash
git clone git@github.com:nikihowe/simplified-reward.git
cd simplified-reward
python3 -m venv venv
python3 -m pip install -r requirements.txt
```

### Implemented environments

There are two environments within the `experiments` directory.

#### Two-state MDP
This is the main environment for which the code was developed. It is a two-state Markov decision process with two actions. It is supported for both (1) and (2).

#### Cleaning robot
This is the first environment presented in the paper. It is an 8-arm bandit with special reward structure. It is supported for (1) and partially for (2).

### Running experiments

See the code in the `experiments` directory for example experiments on the two environments presented above.

1) The `calculate_achievable_permutations` function can be used to calculated which permutations are achievable via a reward function. 
2) The `run_full_ordering_search` function can be used to find all policy orderings which are realizable via some reward function.
3) The `make_ungameability_graph` function can be used to generate a graph of all pairs of policy permutations resulting from the ungameable pairs of reward functions.
3) The `make_simplification_graph` function can be used to generate a graph of all pairs of policy permutations resulting from the simplifications of reward functions.

To run an experiment, modify the code in `experiments` directory as desired, and then modify and call `run.py` from the root directory.