from src.base.analysis.clean_non_core import clean_non_core
from src.spurious_pix.analysis.needed_paths import sp_pix_outputs_dir

if __name__ == "__main__":
    dirs = [sp_pix_outputs_dir]
        
    for dir in dirs:
        clean_non_core(dir)
