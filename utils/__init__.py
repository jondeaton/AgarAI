# Just some general utilities

from utils import rollout

def is_None(x):
    return x is None

def is_not_None(x):
    return x is not None

def none_filter(l):
    return filter(is_not_None, l)