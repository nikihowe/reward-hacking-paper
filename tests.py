import unittest

import numpy as np

from mdp_env import MDPEnv
from policy import Policy, make_cleaning_policy, make_two_state_policy


def two_state_dynamics(state, action):
    del state
    return action


def make_two_state_policies():
    policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
    policy_funs = []

    for policy in policies:
        policy_funs.append(make_two_state_policy(policy))

    return policy_funs


class TestEnvMethods(unittest.TestCase):

    def test_policy_fun_construction(self):
        policies = [(0, 0), (0, 1), (1, 0), (1, 1)]
        policy_funs = []

        for policy in policies:
            policy_funs.append(make_two_state_policy(policy))

        for i, policy_vals in enumerate(policies):
            self.assertEqual(policy_vals[0], policy_funs[i](0))
            self.assertEqual(policy_vals[1], policy_funs[i](1))

    def test_policy_eval(self):
        discount = 0.5
        env = MDPEnv(dynamics=two_state_dynamics, discount=discount)

        policy_funs = make_two_state_policies()
        simple_rewards = np.array([[0, 1],
                                   [2, 2.5]])
        reward_fun = lambda s, a: simple_rewards[s, a]

        all_ave_policy_vals = env.get_all_average_policy_values(policy_permutation=policy_funs, reward_fun=reward_fun)
        np.testing.assert_array_equal(all_ave_policy_vals, np.array([1.0, 2.5, 3.0, 4.25]))

        sorted_policies_and_rewards = env.get_sorted_policies_and_rewards(policies=policy_funs, reward_fun=reward_fun)
        true_policies_and_values = [(make_two_state_policy((0, 0)), 1.0),
                                    (make_two_state_policy((0, 1)), 2.5),
                                    (make_two_state_policy((1, 0)), 3.0),
                                    (make_two_state_policy((1, 1)), 4.25)]
        for i, (policy, val) in enumerate(true_policies_and_values):
            self.assertEqual(sorted_policies_and_rewards[i], (policy, val))

    # Test policies
    def test_policies(self):
        discount = 0.5
        env = MDPEnv(dynamics=two_state_dynamics, discount=discount)

        policy_funs = make_two_state_policies()
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

        policy_funs = make_two_state_policies()
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
