import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from rich import print as print

from composer import global_state

logger = logging.getLogger(__name__)


def set_global_state_information(hydra_conf):
    memory_dir = Path(hydra_conf.protocol.config.root) / "memory"
    memory_dir.mkdir(exist_ok=True, parents=True)
    global_state.set_memory_dir(str(memory_dir))
    logger.info(f"Memory directory set to {global_state.get_memory_dir()}")
    verbosity = HydraConfig.get().verbose
    global_state.set_verbosity(verbosity)
    logger.info(f"Verbose is {global_state.get_verbosity()}")


def run(hydra_conf):
    set_global_state_information(hydra_conf)
    hydra_conf = hydra.utils.instantiate(hydra_conf)
    logger.debug(f"hydra_conf: \n{hydra_conf}")
    for name, target in hydra_conf.protocol.targets.items():
        logger.info(f"Executing: {name}")
        target(hydra_conf)


@hydra.main(version_base="1.3", config_path="configs", config_name="core.yaml")
def hydra_main(hydra_conf):
    """Executes training powered by Hydra, given the configuration file. Note
    that Hydra handles setting up the hydra_conf.

    Parameters
    ----------
    hydra_conf : omegaconf.DictConfig
    """

    run(hydra_conf)


def entrypoint():
    hydra_main()
