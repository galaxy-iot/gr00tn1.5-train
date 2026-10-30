"""Register the robot configuration in Qualcomm's bundled GR00T checkout."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def register_config():
    # Qualcomm imports the pinned repository through its external-repo package.
    from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.experiment.data_config import DATA_CONFIG_MAP
    os.environ['EPISODE_GR00T_PREFIX'] = 'qai_hub_models.models.grootn15.external_repos.gr00t.gr00t'
    sys.path.insert(0, str(ROOT))
    from episode_config import EpisodeJointConfig
    DATA_CONFIG_MAP['episode_pose'] = EpisodeJointConfig()

