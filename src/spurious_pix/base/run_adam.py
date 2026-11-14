import argparse
import logging
from pathlib import Path

from hydra_zen import launch, zen

from src.base.conf_assembly_helpers.artificial_overrides_padder import ArtificialOverridesPadder
from src.spurious_pix.base.override_strings import adam_overrides, autofill_exceptions, intensity_values, muon_overrides

from src.spurious_pix.base.cfg_to_zen_store import ExperimentConfig
from src.train_and_val import train_and_val

import lightning.pytorch as pl

pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_val, pre_call=pre_seed) # Remove seeding here caught as input to config

if __name__ == "__main__":
    """ 
    Via CLI. 
    Fix and pair run/seed (same value for ease) 
    Then slice through the current experiment matches. 
    Here there are 4 experiment ratios to test. 
    """
    log_file = Path(__file__).parent / "logs_run_with_cfg.log"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Script that adds 3 numbers from CMD"
    )
    parser.add_argument("--run", required=False, nargs="?", type=int, help="Enter `run`", default=0)
    parser.add_argument("--seed", required=False, nargs="?", type=int, help="Enter `seed`", default=0)
    parser.add_argument("--slice", required=False, nargs="?", type=int, help="Enter `slice`", default=0)
    parser.add_argument("--injection_seed", required=False, nargs="?", type=int, help="Enter `injection_seed`", default=0)

    args = parser.parse_args()

    injected_run = args.run
    injected_seed = args.seed
    injected_slice = args.slice
    injected_injection_seed = args.injection_seed

    PADDING = len(intensity_values)
    adam_overrides_map = ArtificialOverridesPadder(adam_overrides, autofill_exceptions, PADDING).overrides_map_as_store

    adam_overrides_map.overwrite(
        "run_idx", 
        [f"{injected_run}" for _ in list(range(4))]
    )
    adam_overrides_map.overwrite(
        "seed",
        [f"{injected_seed}" for _ in list(range(4))]
    )
    adam_overrides_map.overwrite(
        "++injection_seed",
        [f"{injected_injection_seed}" for _ in list(range(4))]
    )
    curr_vals = adam_overrides_map.get_map_slice(injected_slice)

    job = launch(
        ExperimentConfig,
        task_function,
        config_name="sp_pix_adam",
        job_name="spurious_pixel",
        version_base="1.3",
        overrides=curr_vals
    )
