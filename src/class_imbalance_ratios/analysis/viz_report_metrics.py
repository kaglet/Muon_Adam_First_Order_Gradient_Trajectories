from src.base.analysis.viz_report_metrics import viz_over_dir
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_l_outputs_dir, cls_imb_s_outputs_dir

if __name__ == "__main__":
    subtype = "small"
    if subtype == "small":
        dirs = [cls_imb_s_outputs_dir]
    else: 
        dirs = [cls_imb_l_outputs_dir]
        
    for dir in dirs:
        viz_over_dir(dir, "multijob/R*")
