"""
File: test
Date: 5/7/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import gym
import gym_agario
from features.extractors import FeatureExtractor

def main():

    env = gym.make("agario-full-v0")

    env.reset()
    extractor = FeatureExtractor()

    for _ in range(100):

        action = (0, 0, 0)
        state, reward, done, info = env.step(action)

        features = extractor(state)

        if done: break

if __name__ == "__main__":
    main()
