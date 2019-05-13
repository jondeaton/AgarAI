"""
File: extractors
Date: 5/7/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
from gym_agario.envs.AgarioFull import Observation


class FeatureExtractor:

    def __init__(self, num_pellet=50, num_virus=5, num_food=10, num_other=5, num_cell=15):
        self.num_pellet = num_pellet
        self.num_virus = num_virus
        self.num_food = num_food
        self.num_other = num_other
        self.num_cell = num_cell

        self.size = 2 * num_pellet + 2 * num_virus + 2 * num_food + 5 * (1 + num_other) * num_cell
        self.filler_value = -1000

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

        pellet_features = self.get_entity_features(loc, observation.pellets, self.num_pellet)
        virus_features =  self.get_entity_features(loc, observation.viruses, self.num_virus)
        food_features =   self.get_entity_features(loc, observation.foods, self.num_food)

        largest_cells = self.largest_cells(agent, n=self.num_cell)
        agent_cell_features = self.get_entity_features(loc, largest_cells, self.num_cell)


        players_features = list()
        closest_players = self.closest_players(loc, observation.others, n=self.num_other)
        for player in closest_players:
            player_cells = self.largest_cells(player, n=self.num_cell)
            player_features = self.get_entity_features(loc, player_cells, self.num_cell)
            players_features.append(player_features)

        # there might not be enough players at all, so just pad the rest
        while len(players_features) < self.num_other:
            empty_features = self.empty_features(self.num_cell, 5)
            players_features.append(empty_features)

        feature_stacks = [pellet_features, virus_features, food_features, agent_cell_features]
        feature_stacks.extend(players_features)

        flattened = list(map(lambda arr: arr.flatten(), feature_stacks))
        features = np.hstack(flattened)
        np.nan_to_num(features, copy=False)
        return features

    def get_entity_features(self, loc, entities, n):
        _, ft_size = entities.shape
        entity_features = np.zeros((n, ft_size))
        close_entities = self.sort_by_proximity(loc, entities, n=n)
        self.to_relative_pos(close_entities, loc)
        num_close, _ = close_entities.shape
        entity_features[:num_close] = close_entities
        return entity_features

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

    def to_relative_pos(self, entities, loc):
        entities[:, (0, 1)] -= loc

    def empty_features(self, n, dim):
        fts = np.ones((n, dim))
        fts[:] = self.filler_value
        return fts

    def position(self, player: np.ndarray):
        # weighted average of cell positions by mass
        if player.size == 0 or player[:, -1].sum == 0:
            return None

        loc = np.average(player[:, (0, 1)], axis=0, weights=player[:, -1])
        return loc
