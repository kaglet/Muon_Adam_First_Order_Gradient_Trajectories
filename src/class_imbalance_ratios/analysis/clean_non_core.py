from src.base.analysis.clean_non_core import clean_non_core
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_l_outputs_dir, cls_imb_s_outputs_dir


if __name__ == "__main__":
    """ 
    Ensure core metrics are kept in .parquet format while 
    non-core e.g. extras, or non-parquet files are cleaned. 
    """
    subtype = "small"
    if subtype == "small":
        dirs = [cls_imb_s_outputs_dir]
    else: 
        dirs = [cls_imb_l_outputs_dir]

    for dir in dirs:
        clean_non_core(dir)