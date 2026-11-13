from src.base.analysis.create_non_core_summaries import summarize_all
from src.spurious_pix.analysis.needed_paths import sp_pix_outputs_dir

if __name__ == "__main__":
    dirs = [sp_pix_outputs_dir]
        
    for dir in dirs:
        summarize_all(dir, "multijob/I*")
