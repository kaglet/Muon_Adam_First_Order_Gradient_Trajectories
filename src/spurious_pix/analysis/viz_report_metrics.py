from src.base.analysis.viz_report_metrics import viz_over_dir
from src.spurious_pix.analysis.needed_paths import sp_pix_outputs_dir

if __name__ == "__main__":
    dirs = [sp_pix_outputs_dir]
        
    for dir in dirs:
        viz_over_dir(dir, "multijob/I*")
