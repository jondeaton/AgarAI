"""
File: extractors
Date: 5/7/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
from gym_agario.envs.AgarioFull import Observation


class FeatureExtractor:

    def __init__(self, nvirus=5, npellets=50, nfood=10, nother=5, ncell=15):
        self.nfood = nfood
        self.nvirus = nvirus
        self.npellets = npellets
        self.nother = nother
        self.ncell = ncell
        self.size = 2 * nfood + 2 * nvirus + 2 * npellets + 5 * ncell + 5 * nother * ncell

    def __call__(self, observation: Observation):
        return self.extract(observation)

    def extract(self, observation: Observation):
        """ extracts features from an observation into a fixed-size feature vector
        :param observation: a named tuple Observation object
        :return: fixed length vector of extracted features about the observation
        """
        agent = observation.agent
        loc = self.position(agent)
        if loc is None: return None # no player position

        close_foods = self.sort_by_proximity(loc, observation.foods, n=self.nfood)
        foods = np.zeros((self.nfood, 2))
        foods[:len(close_foods)] = close_foods

        close_viruses = self.sort_by_proximity(loc, observation.viruses, n=self.nvirus)
        viruses = np.zeros((self.nvirus, 2))
        viruses[:len(close_viruses)] = close_viruses

        close_pellets = self.sort_by_proximity(loc, observation.pellets, n=self.npellets)
        pellets = np.zeros((self.npellets, 2))
        pellets[:len(close_pellets)] = close_pellets

        largest_cells = self.largest_cells(agent, n=self.ncell)
        agent_cells = np.zeros((self.ncell, 5))
        agent_cells[:len(largest_cells)] = largest_cells

        feature_stacks = [foods, viruses, pellets, agent_cells]

        closest_players = self.closest_players(loc, observation.others, n=self.nother)
        for player in closest_players:
            p = np.zeros((self.ncell, 5))
            player_cells = self.largest_cells(player, n=self.ncell)
            p[:len(player_cells)] = player_cells
            feature_stacks.append(p)

        flattened = list(map(lambda arr: arr.flatten(), feature_stacks))
        features = np.hstack(flattened)
        np.nan_to_num(features, copy=False)
        return features


    def largest_cells(self, player, n=None):
        order =  np.argsort(player[:, -1], axis=0)
        return player[order[:n]]

    def closest_players(self, loc, others, n=None):
        distances = list()
        for player in others:
            location = self.position(player)
            distance = np.linalg.norm(loc - location) if location is not None else np.inf
            distances.append(distance)
        order = np.argsort(distances)
        return [others[i] for i in order[:n]]

    def sort_by_proximity(self, loc, entities, n=None):
        positions = entities[:, (0, 1)]
        order = np.argsort(np.linalg.norm(positions - loc, axis=1))
        return entities[order[:n]]

    def position(self, player: np.ndarray):
        # weighted average of cell positions by mass
        if player.size == 0 or player[:, -1].sum == 0:
            return None

        loc = np.average(player[:, (0, 1)], axis=0, weights=player[:, -1])
        return loc