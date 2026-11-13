from src.base.analysis.post_proc_convert import delete_all_parquet_extras
from src.spurious_pix.analysis.needed_paths import sp_pix_outputs_dir

if __name__ == "__main__":
    dirs = [sp_pix_outputs_dir]
    
    for dir in dirs:
        delete_all_parquet_extras(dir)