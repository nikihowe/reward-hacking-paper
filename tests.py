import unittest

import numpy as np

from mdp_env import MDPEnv
from policy import Policy


class TestEnvMethods(unittest.TestCase):

    # Test policies
    def test_policies(self):
        def dynamics(state, action):
            del state
            return action

        discount = 0.5
        env = MDPEnv(dynamics=dynamics, discount=discount)

        policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
        policy_funs = [
            Policy(policies[0], lambda s: policies[0][s]),
            Policy(policies[1], lambda s: policies[1][s]),
            Policy(policies[2], lambda s: policies[2][s]),
            Policy(policies[3], lambda s: policies[3][s]),
        ]
        rewards = np.array([[0, 0],  # state 0
                            [1, 1]])  # state 1
        reward_fun = lambda s, a: rewards[s, a]

        all_ave_policy_vals = env.get_all_average_policy_values(policy_funs, reward_fun)
        np.testing.assert_array_equal(all_ave_policy_vals, np.array([0.5, 1.0, 1.0, 1.5]))

    def test_policies2(self):
        def dynamics(state, action):
            del state
            return action

        discount = 0.5
        env = MDPEnv(dynamics=dynamics, discount=discount)

        policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
        policy_funs = [
            Policy(policies[0], lambda s: policies[0][s]),
            Policy(policies[1], lambda s: policies[1][s]),
            Policy(policies[2], lambda s: policies[2][s]),
            Policy(policies[3], lambda s: policies[3][s]),
        ]
        rewards = np.array([[0, 1],  # state 0
                            [2, 0.5]])  # state 1
        reward_fun = lambda s, a: rewards[s, a]

        all_ave_policy_vals = env.get_all_average_policy_values(policy_funs, reward_fun)
        np.testing.assert_array_equal(all_ave_policy_vals, np.array([1.0, 0.5, 3.0, 1.25]))

    def test_policy_permutation(self):
        def dynamics(state, action):
            del state
            return action

        discount = 0.5
        env = MDPEnv(dynamics=dynamics, discount=discount)

        policies = [(0, 1), (0, 0), (1, 1), (1, 0)]
        policy_funs = [
            Policy(policies[0], lambda s: policies[0][s]),
            Policy(policies[1], lambda s: policies[1][s]),
            Policy(policies[2], lambda s: policies[2][s]),
            Policy(policies[3], lambda s: policies[3][s]),
        ]
        rewards = np.array([[0, 0],  # state 0
                            [1, 1]])  # state 1
        reward_fun = lambda s, a: rewards[s, a]

        all_ave_policy_vals = env.get_all_average_policy_values(policy_funs, reward_fun)
        np.testing.assert_array_equal(all_ave_policy_vals, np.array([1.0, 0.5, 1.5, 1.0]))


if __name__ == '__main__':
    unittest.main()
