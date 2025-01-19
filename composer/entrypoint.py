import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from rich import print as print

from composer import global_state
from composer.utils import Timer

logger = logging.getLogger(__name__)


def set_global_state_information(hydra_conf):
    memory_dir = Path(hydra_conf.protocol.config.root) / "memory"
    memory_dir.mkdir(exist_ok=True, parents=True)
    global_state.set_memory_dir(str(memory_dir))
    logger.debug(f"Memory directory set to {global_state.get_memory_dir()}")
    verbosity = HydraConfig.get().verbose
    global_state.set_verbosity(verbosity)
    logger.debug(f"Verbose is {global_state.get_verbosity()}")


def run_grantgist(hydra_conf):
    with Timer() as global_timer:
        set_global_state_information(hydra_conf)
        hydra_conf = hydra.utils.instantiate(hydra_conf)
        logger.debug(f"hydra_conf: \n{hydra_conf}")
        for name, target in hydra_conf.protocol.targets.items():
            logger.info(f"Executing: {name}")
            with Timer() as timer:
                target(hydra_conf)
            logger.info(f"Completed: {name} in {timer.elapsed:.02f} s")
    logger.info(f"SUCCESS - {global_timer.elapsed:.02f} s")


@hydra.main(version_base="1.3", config_path="configs/grantgist", config_name="core.yaml")
def hydra_main_grantgist(hydra_conf):
    run_grantgist(hydra_conf)


def entrypoint_grantgist():
    hydra_main_grantgist()
