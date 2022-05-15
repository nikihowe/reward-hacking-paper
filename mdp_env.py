# (c) 2022 Nikolaus Howe
import dataclasses
import numpy as np


@dataclasses.dataclass
class MDPEnv(object):
    """
    Used to store some simple MDP elements
    """
    states: np.ndarray
    actions: np.ndarray
    discount: float

    # def get_best_action(state, rewards):
    #     return np.argmax(rewards[state])

    # def get_Q_with_counter(self, state, action, rewards, counter):
    #     if counter > 0:
    #         return rewards[state, action] \
    #                + self.discount * self.get_Q_with_counter(state=self.states[action],
    #                                                          action=get_best_action(state, rewards),
    #                                                          rewards=rewards,
    #                                                          counter=counter - 1)
    #     else:
    #         return 0

    # def get_Q(self, state, action, rewards):
    #     return self.get_Q_with_counter(state=state, action=action, rewards=rewards, counter=100)

    def get_policy_value_with_counter(self, policy, state, reward_fun, counter):
        if counter > 0:
            action = policy[state]
            return reward_fun(state, action) \
                   + self.discount * self.get_policy_value_with_counter(policy=policy,
                                                                        state=self.states[action],
                                                                        reward_fun=reward_fun,
                                                                        counter=counter - 1)
        else:
            return 0

    def get_policy_value(self, policy, state, reward_fun):
        return self.get_policy_value_with_counter(policy=policy,
                                                  state=state,
                                                  reward_fun=reward_fun,
                                                  counter=500)

    def get_average_policy_value(self, policy, reward_fun):
        return (self.get_policy_value_with_counter(policy=policy, state=0, reward_fun=reward_fun, counter=500)
                + self.get_policy_value_with_counter(policy=policy, state=1, reward_fun=reward_fun, counter=500)) / 2

    def get_all_average_policy_values(self, policy_permutation, reward_fun):
        res = []
        for policy in policy_permutation:
            res.append(self.get_average_policy_value(policy, reward_fun))
        return res

    def get_sorted_policies_and_rewards(self, policies, reward_fun):
        policies_and_rewards = []
        for policy in policies:
            value_of_policy = self.get_average_policy_value(policy, reward_fun)
            policies_and_rewards.append((policy, value_of_policy))
        sorted_policies_and_rewards = sorted(policies_and_rewards, key=lambda x: x[-1])
        return sorted_policies_and_rewards

    def get_ineqs_from_policies_and_rewards(self, policies, reward_fun):
        sorted_policies_and_rewards = self.get_sorted_policies_and_rewards(policies=policies, reward_fun=reward_fun)

        inequalities = []
        for i, policy in enumerate(sorted_policies_and_rewards):
            if i == 0:
                continue
            inequalities.append((sorted_policies_and_rewards[i - 1][0], policy[0]))
        #   print(inequalities)
        return inequalities
