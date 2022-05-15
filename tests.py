import unittest

import numpy as np

from mdp_env import MDPEnv


class TestEnvMethods(unittest.TestCase):

    # Test policies
    def test_policies(self):
        states = np.array([0, 1])
        actions = np.array([0, 1])
        discount = 0.5
        env = MDPEnv(states=states, actions=actions, discount=discount)

        policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
        rewards = np.array([[0, 0],  # state 0
                            [1, 1]])  # state 1
        reward_fun = lambda s, a: rewards[s, a]

        all_ave_policy_vals = env.get_all_average_policy_values(policies, reward_fun)
        np.testing.assert_array_equal(all_ave_policy_vals, np.array([0.5, 1.0, 1.0, 1.5]))

    def test_policies2(self):
        states = np.array([0, 1])
        actions = np.array([0, 1])
        discount = 0.5
        env = MDPEnv(states=states, actions=actions, discount=discount)

        policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
        rewards = np.array([[0, 1],  # state 0
                            [2, 0.5]])  # state 1
        reward_fun = lambda s, a: rewards[s, a]

        all_ave_policy_vals = env.get_all_average_policy_values(policies, reward_fun)
        np.testing.assert_array_equal(all_ave_policy_vals, np.array([1.0, 0.5, 3.0, 1.25]))

    def test_policy_permutation(self):
        states = np.array([0, 1])
        actions = np.array([0, 1])
        discount = 0.5
        env = MDPEnv(states=states, actions=actions, discount=discount)

        policies = [(0, 1), (0, 0), (1, 1), (1, 0)]
        rewards = np.array([[0, 0],  # state 0
                            [1, 1]])  # state 1
        reward_fun = lambda s, a: rewards[s, a]

        all_ave_policy_vals = env.get_all_average_policy_values(policies, reward_fun)
        np.testing.assert_array_equal(all_ave_policy_vals, np.array([1.0, 0.5, 1.5, 1.0]))

    # Test

if __name__ == '__main__':
    unittest.main()
