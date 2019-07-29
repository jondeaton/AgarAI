"""
File: test
Date: 2019-07-28 
Author: Jon Deaton (jdeaton@stanford.edu)
"""


def test(model, env, render=True):
    obs, done, ep_reward = env.reset(), False, 0
    while not done:
        action, _ = model.action_value(obs[None, :])
        obs, reward, done, _ = env.step(action)
        ep_reward += reward
        if render:
            env.render()
    return ep_reward


def main():
    pass


if __name__ == "__main__":
    main()
