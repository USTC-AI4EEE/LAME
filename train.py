from lightning.pytorch.cli import LightningCLI

from DRCT import WindSRDRCT
from weatherdata import WeatherDataModule

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.getLogger('multiprocessing.util').setLevel(logging.CRITICAL)


def cli_main():
    LightningCLI(
        model_class=WindSRDRCT,
        datamodule_class=WeatherDataModule,
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main()
