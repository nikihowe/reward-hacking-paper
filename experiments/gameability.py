# (c) 2022 Nikolaus Howe
import numpy as np

from itertools import combinations


# Some single-use gameability testing code for the cleaning robot domain
def check_reward_gameability(holistic_reward_fun, narrow_reward_fun):
    """
    Check if the proxy reward function is gameable with respect to the
    true reward function.
    """

    all_actions = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)]

    counter = 0
    for action1, action2 in combinations(all_actions, 2):
        counter += 1
        if holistic_reward_fun(action1) < holistic_reward_fun(action2) and \
                narrow_reward_fun(action2) < narrow_reward_fun(action1):
            print(f"The true reward function says that {action2}: "
                  f"{holistic_reward_fun(action2)} is better than {action1}: {holistic_reward_fun(action1)}")
            print(f"but the proxy reward function says that {action1}: "
                  f"{holistic_reward_fun(action1)} is better than {action2}: {narrow_reward_fun(action2)}")
            return False

    print("checked", counter)
    return True


if __name__ == "__main__":
    def hr1(action):
        return action @ np.array([3, 4, 5])


    def nr1(action):
        return action @ np.array([4, 4, 4])


    def hr2(action):
        return action @ np.array([5, 2, 2])


    def nr2(action):
        return action @ np.array([1, 0, 0])


    def nr3(action):
        return action @ np.array([2, 2, 2])


    def hr3(action):
        return action @ np.array([1, 0, 0])


    print(check_reward_gameability(hr1, nr1))
    print(check_reward_gameability(hr2, nr2))
    print(check_reward_gameability(hr3, nr3))
