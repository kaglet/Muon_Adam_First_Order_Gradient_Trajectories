import logging
import os
from pathlib import Path
import sys
# submit_grid.py
import subprocess
import os
import time
from typing import Tuple

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
    retry_counter = 0

    num_optims = len(optim_names)
    run_length = len(run_interval)
    slice_length = len(slice_interval)
    run = 0
    while run < run_length:
        slice = 0
        while slice < slice_length:
            optim_idx = 0
            while optim_idx < num_optims:
                injected_run: int = run_interval[run]
                injected_slice: int = slice_interval[slice]
                optim_name = optim_names[optim_idx]

                print(f"Generating and submitting job for {optim_name} run={injected_run}, slice={injected_slice}")

                with open(root_dir.resolve() / f"run_{optim_name}.txt", "r") as f:
                    template_content = f.read()

                job_script = template_content
                job_script = job_script.replace("{{run}}", str(injected_run)).replace("{{seed}}", str(injected_run))
                job_script = job_script.replace("{{slice}}", str(injected_slice)).replace("{{ratio_seed}}", str(injected_slice))
                
                temp_filename = "temp_generated_run.txt"
                with open(temp_filename, "w") as f:
                    f.write(job_script)

                logging.debug("=== LOGGING TXT SCRIPT ===")
                logging.debug(job_script)

                # Launch TXT script
                attempt = subprocess.run(
                    ["python", temp_filename],
                    text=True, 
                    shell=True, 
                    capture_output=True, 
                )

                if attempt.returncode == 0:
                    msg = attempt.stdout.lower()
                else:
                    msg = attempt.stderr.lower()
                    fatal_detected = react_to_fatal_err(msg=msg)
                    if fatal_detected:
                        print(f"FATAL: Unexpected cluster error: {attempt.stderr}")
                        sys.exit(1)

                SLEEP_TIME = 60 * 60 # (sec x hr = 1hr default)
                reaction = react_to_limits(msg=msg, retry_counter=retry_counter, optim_idx=optim_idx, SLEEP_TIME=SLEEP_TIME)
                retry_counter = reaction["retry_counter"]
                optim_idx = reaction["optim_idx"]

                if retry_counter > MAX_RETRIES_IDX:
                    print(f"FATAL: Reached max retry limit idx: {MAX_RETRIES_IDX}. Exiting...")
                    sys.exit(1)

            slice += 1

        run += 1

if __name__ == "__main__":
    dirs = [Path(__file__).parent.parent / "small/", Path(__file__).parent.parent / "large/"]

    for root_dir in dirs:
        print(str(root_dir))
        runs = list(range(10))
        # runs = runs[3:]
        slices = list(range(3))

        optim_names = ["adam", "muon"]
        run_local_many(optim_names=optim_names, root_dir=root_dir, run_interval = runs, slice_interval = slices)