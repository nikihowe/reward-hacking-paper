# (c) 2022 Nikolaus Howe
import dataclasses
import numpy as np

from typing import Callable

from policy import Policy

POLICY_EVAL_HORIZON = 200  # how far in the future to calculate discounted rewards


@dataclasses.dataclass
class MDPWithoutRewardEnv(object):
    """
    Used to store some simple MDP elements
    """
    dynamics: Callable
    discount: float
    num_states: int = 2
    num_actions: int = 2
    require_nonnegative_reward: bool = False

    # def get_discounted_state_action_occupancy(self, state: int, ):
    #     occupancy_matrix = np.zeros((self.num_states, self.num_actions))

    def get_policy_value_with_counter(self, state: int, policy_fun: Policy, reward_fun: Callable, counter):
        if counter > 0:
            action = policy_fun(state)
            # print("state", state)
            # print("action", action)
            # print("reward_fun", reward_fun(state, action))
            return reward_fun(state, action) \
                   + self.discount * self.get_policy_value_with_counter(policy_fun=policy_fun,
                                                                        state=self.dynamics(state=state, action=action),
                                                                        reward_fun=reward_fun,
                                                                        counter=counter - 1)
        else:
            return 0

    def get_policy_value(self, policy_fun, state, reward_fun):
        if self.discount == 0:  # single step case
            return reward_fun(state, policy_fun(state))
        else:
            return self.get_policy_value_with_counter(state=state,
                                                      policy_fun=policy_fun,
                                                      reward_fun=reward_fun,
                                                      counter=POLICY_EVAL_HORIZON)

    def get_average_policy_value(self, policy_fun, reward_fun):
        total = 0
        for i in range(self.num_states):
            total += self.get_policy_value_with_counter(policy_fun=policy_fun,
                                                        state=i,
                                                        reward_fun=reward_fun,
                                                        counter=POLICY_EVAL_HORIZON)
        return total / self.num_states

    def get_all_average_policy_values(self, policy_permutation: tuple[Policy],
                                      reward_fun: Callable[[int, int], float]):
        res = []
        for policy_fun in policy_permutation:
            res.append(self.get_average_policy_value(policy_fun, reward_fun))
        return res

    def get_sorted_policies_and_rewards(self, policies: tuple[Policy], reward_fun: Callable[[int, int], float]):
        policies_and_rewards = []
        for policy_fun in policies:
            value_of_policy = self.get_average_policy_value(policy_fun, reward_fun)
            policies_and_rewards.append((policy_fun, value_of_policy))
        sorted_policies_and_rewards = sorted(policies_and_rewards, key=lambda x: x[-1])
        return sorted_policies_and_rewards

    def get_ineqs_from_policies_and_rewards(self, policies: tuple[Policy], reward_fun):
        sorted_policies_and_rewards = self.get_sorted_policies_and_rewards(policies=policies, reward_fun=reward_fun)

        inequalities = []
        for i, policy in enumerate(sorted_policies_and_rewards):
            if i == 0:
                continue
            inequalities.append((sorted_policies_and_rewards[i - 1][0], policy[0]))
        #   print(inequalities)
        return inequalities
