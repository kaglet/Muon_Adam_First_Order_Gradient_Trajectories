import logging
import os
from pathlib import Path
import sys
# submit_grid.py
import subprocess
import os
import time
from typing import Tuple

from src.class_imbalance_ratios.small.override_strings import downsampler_ratio_values as small_ratios
from src.spurious_pix.base.override_strings import intensity_values

my_log_file = Path(__file__).parent / "logs/edit_run_txt.log"
os.makedirs(my_log_file.parent, exist_ok=True)
with open(my_log_file, "a") as file:
    pass

logging.basicConfig(filename=my_log_file, level=logging.DEBUG)

def react_to_limits(msg: str, retry_counter: int, optim_idx: int, SLEEP_TIME: int) -> dict[str, int]:
    """
    Advance loop if no limits detected, else keep loop stagnant. 
    Reset retry counter due to none busy cluster, else advance the counter to prevent infinite loop. 
    """
    if "exceed" in msg or "limit" in msg:
        time.sleep(SLEEP_TIME)
        retry_counter += 1
        optim_idx = optim_idx
    else:
        retry_counter = 0
        optim_idx += 1

    return dict(retry_counter=retry_counter, optim_idx=optim_idx)

def react_to_fatal_err(msg: str) -> bool:
    if (not "exceed" in msg) and (not "limit" in msg):
        return True

    return False

def run_local_many(optim_names: list[str], root_dir: Path, run_interval: list[int], slice_interval: list[int], MAX_RETRIES_IDX: int = 100):
    for injected_run in run_interval:
        for injected_slice in slice_interval:
            for optim_name in optim_names:
                print(f"Generating and submitting job for {optim_name} run={injected_run}, slice={injected_slice}")

                template_path = root_dir.resolve() / f"run_{optim_name}.txt"
                with open(template_path, "r", encoding="utf-8") as f:
                    template_content = f.read()

                # Clean whitespace/newlines and perform string injections
                job_cmd = template_content.strip()
                job_cmd = job_cmd.replace("{{run}}", str(injected_run)).replace("{{seed}}", str(injected_run))
                job_cmd = job_cmd.replace("{{slice}}", str(injected_slice)).replace("{{injection_seed}}", str(injected_slice))

                logging.debug("=== EXECUTING SCRIPT COMMAND ===")
                logging.debug(job_cmd)

                # Ensure output streams directly to console without hanging
                attempt = subprocess.run(
                    job_cmd,
                    shell=True,
                    text=True,
                    check=False
                )

                if attempt.returncode != 0:
                    print(f"FAILED: {optim_name} run={injected_run}, slice={injected_slice} returned code {attempt.returncode}")
                else:
                    print(f"COMPLETED: {optim_name} run={injected_run}, slice={injected_slice}")

if __name__ == "__main__":
    dirs = [Path(__file__).parent.parent / "base/"] # Only one subtype = "base" to source optim scripts for
    slices = list(range(len(intensity_values))) 

    runs = list(range(0,10)) # TODO: Allow this to be changed from terminal
    optim_names = ["adam", "muon"]
    for root_dir in dirs:
        print(str(root_dir))
        run_local_many(optim_names=optim_names, root_dir=root_dir, run_interval = runs, slice_interval = slices)