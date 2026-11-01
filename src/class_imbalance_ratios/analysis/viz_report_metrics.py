import os
from pathlib import Path
from typing import Any, Type

from types import SimpleNamespace
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_s_outputs_dir, cls_imb_l_outputs_dir
from src.class_imbalance_ratios.analysis.plot_managers import GridPlotOptions, GridPlotsmanager, LonePlotOptions, LonePlotManager, PlotManager, PlotOptions

def get_mix_match_names(mode: str, final_fig_name: str, experim_dir: Path, path: Path, v_dir: Path) -> SimpleNamespace:
    optimizer_name = "Muon" if "muon" in str(experim_dir).lower() else "Adam"

    last_underscore = str(path.stem).rfind("_") + 1
    version = str(v_dir.stem).upper()
    metric_name = str(path.stem).upper()[last_underscore: ].replace("-", " ").replace("_", " ")

    x_label = "Epochs" if "epochs" in path.stem.lower() else "Steps" # Default is steps assuming no aggregation
    y_label = "Unknown"
    if "accuracy" in path.stem.lower():
        y_label = "Accuracy" 
    elif "loss" in path.stem.lower():
        y_label = "Loss" 
    elif "unnormalised" in path.stem.lower():
        y_label = "Dot Product"
    else:
        y_label = "Cosine Similarity"

    if "train" in path.stem.lower():
        train_split = "train"
    elif "val" in path.stem.lower(): 
        train_split = "val"
    else: 
        train_split = "train"
    
    train_split = train_split.title()

    optimizer_name = optimizer_name.title()
    mode = mode.title()
    final_fig_name=final_fig_name.title()
    x_label = x_label.title()
    y_label = y_label.title()
    mix_match_names = SimpleNamespace(use_prop_cycle=cycle_on_prop_checker(path), mode=mode, unique_cols=use_unique_col_names(path), version=version, metric_name=metric_name, x_label = x_label, y_label = y_label, final_fig_name=final_fig_name, optimizer_name=optimizer_name, train_split=train_split)

    return mix_match_names

def cycle_on_prop_checker(path: Path) -> bool:
    cycle_yes = True
    if "accuracy" in path.stem.lower():
        if "subgroup_steps" in str(path.stem).lower(): # subgroup cases first
            cycle_yes = False
        elif "subgroup_epochs" in str(path.stem).lower():
            cycle_yes = False
        elif "epochs" in path.stem.lower(): # fine grained per epoch case, no subgroups detected
            cycle_yes = False 

    return cycle_yes

def use_unique_col_names(path: Path) -> bool:
    unique_cols = False
    if "accuracy" in path.stem.lower():
        if "subgroup_steps" in str(path.stem).lower(): # subgroup cases first
            unique_cols = True
        elif "subgroup_epochs" in str(path.stem).lower():
            unique_cols = True
        elif "epochs" in path.stem.lower(): # fine grained per epoch case, no subgroups detected
            unique_cols = True 

    return unique_cols

def viz_n_metrics(plot_options_type: Type[PlotOptions], plot_manager_type: Type[PlotManager], mode:str, outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, final_fig_name:str, plot_cycler_settings: dict | None) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    fig_ax_n_store: list = []
    plots_managers = []

    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob(version_dir))
        
        for i, v_dir in enumerate(version_dirs):
            # print(f"v_dir: {v_dir.stem} matched to idx: {i}")
            # Better for figax_store to have a fixed length based of versions
            # Then index into that when repeating diff experiments
            start_overwrite_figaxs = len(fig_ax_n_store) == len(version_dirs)
            if not start_overwrite_figaxs:
                fig_ax_n_store.append(None)

            for info in run_paths_info:
                path = sorted(v_dir.rglob(info["relative_path"], case_sensitive=False))[0]
                if not Path(path).exists():
                    raise LookupError(f"Path to requested metric: {path} does not exit.")

                metric_df = pd.read_parquet(path)

                names = get_mix_match_names(mode, final_fig_name, experim_dir, path, v_dir)
                metric_df.rename(columns={"Filler":names.metric_name}, inplace=True)

                options = plot_options_type(plot_name="lone", plot_cycler=plot_cycler_settings, use_cols=names.unique_cols, version=names.version, train_split=names.train_split, final_fig_name=names.final_fig_name, mode=names.mode, optim_name=names.optimizer_name, base_size=8, x_label=names.x_label, y_label=names.y_label)
                plots_manager = plot_manager_type(metric_df=metric_df, plt_options=options)

                if not start_overwrite_figaxs and fig_ax_n_store[i] is None:
                    fig_ax_n_store[i] = plots_manager # For runs in info they transfer on the same figax
                else:
                    used_manager = fig_ax_n_store[i]
                    assert type(used_manager) is plot_manager_type, ""
                    plots_manager.overwrite_fig_ax_n(fig_ax_n=used_manager.fig_ax_n)

                plots_managers.append(plots_manager)

    for i, mngr in enumerate(plots_managers):
        assert type(mngr) is PlotManager
        fig, ax_n = mngr.viz_cols_in_ax_n()
        save_dir = f"{str(outputs_dir.resolve())}/report/{mode.lower()}/{mngr.plot_option_names["version"]}/"
        os.makedirs(save_dir, exist_ok=True)
        # Directory path should ensure required uniqueness settings 
        assert type(fig) is Figure, "Given fig is not of type Figure"
        fig.savefig(f"{save_dir}/{mngr.plot_option_names["plot_type_name"]}_{final_fig_name.replace(" ", "_").lower()}.png") 

def viz_lone_metrics( mode:str, outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, final_fig_name:str, plot_cycler_settings: dict | None) -> None:
    viz_n_metrics(plot_options_type=LonePlotOptions, plot_manager_type=GridPlotsmanager, plot_cycler_settings=plot_cycler_settings, mode=mode, outputs_dir=outputs_dir, run_paths_info=run_paths_info, version_dir=version_dir, final_fig_name=final_fig_name)

def viz_grid_metrics(mode:str, outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, final_fig_name:str, plot_cycler_settings: dict | None) -> None:
    viz_n_metrics(plot_options_type=LonePlotOptions, plot_manager_type=LonePlotManager, plot_cycler_settings=plot_cycler_settings, mode=mode, outputs_dir=outputs_dir, run_paths_info=run_paths_info, version_dir=version_dir, final_fig_name=final_fig_name)

def viz_mean_correlation(dir_path: Path, mode: str) -> None:
    print(f"=========== VIZ {mode.upper()} NORMED GRAD CORR METRICS FOR REPORT ===========") 
    # Normed Per col per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-none_{mode}_Normalised-GC.parquet", 
        )
    ]

    NUM_PROPS_IN_CYCLE = 2
    plot_cycler_settings = dict(color=['#CC79A7', '#0072B2'], linestyle=['--', '--'], marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
    
    viz_grid_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info,
        final_fig_name="Muon and Adam Per Layer Cosine Similarity"
    ) 

    # Normed Col reduction per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-cols_{mode}_Normalised-GC.parquet", 
        )
    ]

    viz_lone_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info,
        final_fig_name="Muon and Adam Cosine Similarity"
    ) 

def viz_dot_prod(dir_path: Path, mode: str) -> None:
    print(f"=========== VIZ {mode.upper()} UNNORMED GRAD CORR METRICS FOR REPORT ===========") 
    # Unnormed Per col per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-none_{mode}_Unnormalised-GC.parquet", 
        )
    ]

    NUM_PROPS_IN_CYCLE = 2
    plot_cycler_settings = dict(color=['#CC79A7', '#0072B2'], linestyle=['--', '--'], marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
        
    viz_grid_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info,
        final_fig_name="Muon and Adam Per Layer Dot Product"
    ) 

    # Unnormed Col reduction per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-cols_{mode}_Unnormalised-GC.parquet", 
        )
    ]

    viz_lone_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info,
        final_fig_name="Muon and Adam Dot Product"
    ) 

def viz_accuracy(dir_path: Path, summary_mode: str) -> None:
        print(f"=========== VIZ {summary_mode.upper()} SUMMARY ACCURACY METRICS FOR REPORT ===========") 
        # Cols present per epoch
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-epochs_{summary_mode}_Train-Accuracy.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-epochs_{summary_mode}_Validation-Accuracy.parquet", 
            ) 
        ]

        plot_cycler_settings = None
        viz_lone_metrics(
            plot_cycler_settings=plot_cycler_settings,
            mode=summary_mode,
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Per Class Muon and Adam Train and Validation Accuracy"
        ) 
        
        # Col reduction per epoch
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-both_{summary_mode}_Train-Accuracy.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-both_{summary_mode}_Validation-Accuracy.parquet", 
            ) 
        ]

        NUM_PROPS_IN_CYCLE = 4
        plot_cycler_settings = dict(color=['#CC79A7', '#CC79A7', '#0072B2', '#0072B2'], linestyle=['-', '--'] * int(NUM_PROPS_IN_CYCLE / 2), marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
        viz_lone_metrics(
            plot_cycler_settings=plot_cycler_settings,
            mode=summary_mode,
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Accuracy (Global)"
        ) 

        # Subgroups per epoch
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-subgroup_epochs_{summary_mode}_Train-Accuracy.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-subgroup_epochs_{summary_mode}_Validation-Accuracy.parquet", 
            ) 
        ]

        NUM_PROPS_IN_CYCLE = 8
        # Cycle is (Adam Majority, Adam Minority) Train then Val --- (Muon Majority, Muon Minority) Train then Val
        plot_cycler_settings = dict(color=['#CC79A7', '#E69F00', '#CC79A7', '#E69F00', '#0072B2', '#56B4E9', '#0072B2', '#56B4E9'], linestyle=['-', '-', '--', '--'] * 2, marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)

        viz_lone_metrics(
            plot_cycler_settings=plot_cycler_settings,
            mode=summary_mode,
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Subgroup Muon and Adam Train and Validation Accuracy"
        ) 

def viz_loss(dir_path: Path, mode: str) -> None:
        print(f"=========== VIZ {mode.upper()} SUMMARY LOSS METRICS FOR REPORT ===========") 
        # Loss per step (NOT per optim step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-none_{mode}_Train-Loss.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-none_{mode}_Validation-Loss.parquet", 
            ) 
        ]

        NUM_PROPS_IN_CYCLE = 4
        plot_cycler_settings = dict(color=['#CC79A7', '#CC79A7', '#0072B2', '#0072B2'], linestyle=['-', '--'] * int(NUM_PROPS_IN_CYCLE / 2), marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
        viz_lone_metrics(
            plot_cycler_settings=plot_cycler_settings,
            mode=mode,
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Loss"
        ) 

def viz_over_dir(dir_path: Path) -> None:
    """ 
    Visualize non-core metrics that were created. 
    """
    SUMMARY_MODES: list[str] = ["mean", "std", "upside_dev", "downside_dev", "skewness"]
    subset_summary_modes = SUMMARY_MODES[2:] # Generate only a subset of modes especially if new modes are added in @./create_non_core_summaries.py
    for mode in subset_summary_modes: 
        viz_accuracy(dir_path=dir_path, summary_mode=mode)
        viz_loss(dir_path=dir_path, mode=mode)
        viz_dot_prod(dir_path=dir_path, mode=mode)
        viz_mean_correlation(dir_path=dir_path, mode=mode)

if __name__ == "__main__":
    viz_over_dir(cls_imb_s_outputs_dir)