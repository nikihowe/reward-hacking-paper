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
2) The `run_full_simplification_search` function can be used to attempt to find reward functions which satisfy a given policy ordering simplification. In addition to specifying the policies to be equated, the user can further specify whether or not to impose equality, strict inequality, or inclusive inequality on policies which are adjacent in the initial ordering.