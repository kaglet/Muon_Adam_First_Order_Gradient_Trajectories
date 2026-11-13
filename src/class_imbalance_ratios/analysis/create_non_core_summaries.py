from src.base.analysis.create_non_core_summaries import summarize_all
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_l_outputs_dir, cls_imb_s_outputs_dir

if __name__ == "__main__":
    subtype = "small"
    if subtype == "small":
        dirs = [cls_imb_s_outputs_dir]
    else: 
        dirs = [cls_imb_l_outputs_dir]
        
    for dir in dirs:
        summarize_all(dir, "multijob/R*")
