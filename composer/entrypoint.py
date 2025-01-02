import logging

import hydra

# from composer import global_state

logger = logging.getLogger(__name__)


def run(hydra_conf):
    # global_state.set_memory_dir(hydra_conf.paths.memory)
    hydra_conf = hydra.utils.instantiate(hydra_conf)
    logger.debug(f"hydra_conf: \n{hydra_conf}")
    hydra_conf.protocol.obj(hydra_conf).run()


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
