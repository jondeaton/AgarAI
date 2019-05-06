"""
File: __init__.py
Date: 5/6/19 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import sys
import logging
from log.tensorboard import Tensorboard

tensorboard = Tensorboard.getInstance(None)