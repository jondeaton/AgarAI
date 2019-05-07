"""
File: Configuration
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import configparser


class Configuration(object):
    """
    A class simply for holding configurations (i.e. paths to data files, etc.)
    The idea here is that the
    """

    def __init__(self, config_file):
        # Setup the filesystem configuration
        self._config_file = os.path.join(config_file)
        self._config = configparser.ConfigParser()
        self._config.read(self._config_file)
        c = self._config

        self.path = os.path.expanduser(c["Data"]["path"])

    def override(self, settings):
        """
        Allows for variables in the configuration to be over-ridden.
        All attributes of "settings" which are also attributes
        of this object will be set to the values found in "settings"
        :param settings: Object with attributes to override in this object
        :return: None
        """
        for attr in vars(settings):
            if getattr(self, attr) is not None:
                value = getattr(settings, attr)
                setattr(self, attr, value)