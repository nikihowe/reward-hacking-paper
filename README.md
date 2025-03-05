# [Defining and Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085) accompanying code

<p align="center">
  <img src="https://github.com/user-attachments/assets/c99044c5-687a-401b-a696-6813a3c8a1ff" width="500">
</p>

## Installation

```bash
git clone https://github.com/nikihowe/reward-hacking-paper.git
cd reward-hacking-paper
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Implemented environments

There are two environments within the `experiments` directory.

### Two-state MDP
This is the main environment for which the code was developed. It is a two-state Markov decision process with two actions. It is supported for both (1) and (2).

### Cleaning robot
This is the first environment presented in the paper. It is an 8-arm bandit with special reward structure. It is supported for (1) and partially for (2).

## Running experiments

See the code in the `experiments` directory for example experiments on the two environments presented above.

1) The `calculate_achievable_permutations` function can be used to calculated which permutations are achievable via a reward function. 
2) The `run_full_ordering_search` function can be used to find all policy orderings which are realizable via some reward function.
3) The `make_ungameability_graph` function can be used to generate a graph of all pairs of policy permutations resulting from the ungameable pairs of reward functions.
3) The `make_simplification_graph` function can be used to generate a graph of all pairs of policy permutations resulting from the simplifications of reward functions.

To run an experiment, modify the code in `experiments` directory as desired, and then modify and call `run.py` from the root directory.

## Citation

```
@article{skalse2022defining,
  title={Defining and characterizing reward gaming},
  author={Skalse, Joar and Howe, Nikolaus and Krasheninnikov, Dmitrii and Krueger, David},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9460--9471},
  year={2022}
}
```
