"""General utilities."""

import os


def get_training_dir(output_dir: str, name: str) -> str:
    """Find a suitable training directory.

    Finds a suitable subdirectory within `output_dir` to save
    files from this run named `name`.

    :param output_dir: Global output directory.
    :param name: Name of this run.
    :return: Path to file of the form /path/to/output/name-X
    """
    base = os.path.join(output_dir, name)
    if not os.path.exists(base):
        return base
    i = 1
    while os.path.exists(f"{base}-{i:03d}"):
        i += 1
    return f"{base}-{i:03d}"