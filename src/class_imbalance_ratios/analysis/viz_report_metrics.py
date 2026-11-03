import os
from pathlib import Path
from typing import Any, Optional, Tuple, Type

from types import SimpleNamespace
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_s_outputs_dir, cls_imb_l_outputs_dir
from src.class_imbalance_ratios.analysis.plot_managers import GridPlotOptions, GridPlotManager, LonePlotOptions, LonePlotManager, PlotManager, PlotOptions

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

def create_experiment_plot_managers(plot_options_type: Type[PlotOptions], plot_manager_type: Type[PlotManager], mode:str, outputs_dir: Path, metric_path_info: dict[str, Any], version_dir_pattern: str, final_fig_name:str, plot_cycler_settings: dict | None, fig_ax_n_store: Optional[list[Any]] = None) -> Tuple[list[Any], list[Any]]:
    """ 
    Given a directory containing experiments over optimisers does the following:
    (1) Creates the same figure and axes per (optimiser,version) combo. This allows re-use of the same object. 
        (1.2) Re-use of the (figure,axes) is done for metrics at file path pattern desired from the version directory. 
            - Therefore plot sharing happens twice. 
            ~ Alternatively this could happen one after the other instead of another inner loop. 
            Therefore accept only one summary_mode, one metric path we are seeking. 
    (2) Create a unique plot manager per (optimiser,version,data) combo. 
    That is, for an (optimiser,version) combo, when e.g. train then validation data is requested via 
    the file path pattern desired, a new plot manager is created. 
        (2.1) 
    (3) At end of plot managers creation, saves the visualised metrics. 
    ~ Hence the logic of populating and visualising content of plot managers can be decoupled. 
    ~ Start with info, assuming size 1. 
    ~ Make fixed sized plot managers prepopulated even (or perhaps this is exactly where it occurs). Then do overwrite loop. 
    """
    # There will be as many plots as there are versions * optims, 1 per optim and version
    # Each plot manager keeps state data only for optim and version
    optims_dirs = sorted(outputs_dir.iterdir(), reverse=True)
    plots_managers = []
    fig_ax_n_store = [] if fig_ax_n_store is None else fig_ax_n_store

    for optim_idx, experim_dir in enumerate(optims_dirs):
        if "report" in str(experim_dir):
            continue
        version_dirs = sorted(experim_dir.glob(version_dir_pattern))
        if len(fig_ax_n_store) > 0:
            assert len(fig_ax_n_store) == len(version_dirs), f"Lengths of figures vs versions for experiment do not match: {len(fig_ax_n_store)} != {len(version_dirs)}"
        should_create_new = len(fig_ax_n_store) != len(version_dirs)
        for version_idx, version_dir in enumerate(version_dirs):
            metric_path = sorted(version_dir.rglob(metric_path_info["relative_path"], case_sensitive=False))[0]
            if not Path(metric_path).exists():
                raise LookupError(f"Path to requested metric: {metric_path} does not exit.")

            metric_df = pd.read_parquet(metric_path)

            names = get_mix_match_names(mode, final_fig_name, experim_dir, metric_path, version_dir)
            metric_df.rename(columns={"Filler":names.metric_name}, inplace=True)

            QUANTILES = ["q50", "q25", "q75", "q10", "q90", "q5", "q95"]
            quantile_dfs = dict()
            for q in QUANTILES:
                q_path = str(metric_path).replace("mean", q)
                q_df = pd.read_parquet(q_path)
                q_df.rename(columns={"Filler":names.metric_name}, inplace=True)
                quantile_dfs[q] = q_df
                
            options = plot_options_type(plot_cycler=plot_cycler_settings, use_cols=names.unique_cols, version=names.version, train_split=names.train_split, final_fig_name=names.final_fig_name, mode=names.mode, optim_name=names.optimizer_name, base_size=8, x_label=names.x_label, y_label=names.y_label)
            plots_manager = plot_manager_type(metric_df=metric_df, plt_options=options, quantile_dfs=quantile_dfs)

            if should_create_new: 
                fig_ax_n_store.append(plots_manager.fig_ax_n)
            else:
                fig_ax_n = fig_ax_n_store[version_idx]
                plots_manager.overwrite_fig_ax_n(fig_ax_n=fig_ax_n)

            plots_managers.append(plots_manager)
    
    return plots_managers, fig_ax_n_store

def viz_plot_managers(plots_managers: list[PlotManager], outputs_dir: Path):
    for i, mngr in enumerate(plots_managers):
        assert isinstance(mngr, PlotManager), "Manager is not of required type PlotManager"
        fig, ax_n = mngr.viz_cols_in_ax_n()
        save_dir = f"{str(outputs_dir.resolve())}/report/{mngr.plot_option_names["mode"].lower()}/{mngr.plot_option_names["version"]}/"
        os.makedirs(save_dir, exist_ok=True)
        # Directory path should ensure required uniqueness settings 
        assert type(fig) is Figure, "Given fig is not of type Figure"
        fig.savefig(f"{save_dir}/{mngr.plot_option_names["plot_type_name"]}_{mngr.plot_option_names["final_fig_name"].replace(" ", "_").lower()}.png") 

def manage_lone_metrics(mode:str, outputs_dir: Path, metric_path_info: dict[str, Any], version_dir_pattern: str, final_fig_name:str, plot_cycler_settings: dict | None, fig_ax_n_store: Optional[list[Any]] = None) -> Tuple[list[Any], list[Tuple[Figure, Any]]]:
    plot_managers, fig_ax_n_store = create_experiment_plot_managers(fig_ax_n_store=fig_ax_n_store, plot_options_type=LonePlotOptions, plot_manager_type=LonePlotManager, plot_cycler_settings=plot_cycler_settings, mode=mode, outputs_dir=outputs_dir, metric_path_info=metric_path_info, version_dir_pattern=version_dir_pattern, final_fig_name=final_fig_name)
    return plot_managers, fig_ax_n_store

def manage_grid_metrics(mode:str, outputs_dir: Path, metric_path_info: dict[str, Any], version_dir_pattern: str, final_fig_name:str, plot_cycler_settings: dict | None, fig_ax_n_store: Optional[list[Any]] = None) -> Tuple[list[Any], list[Tuple[Figure, Any]]]:
    plot_managers, fig_ax_n_store = create_experiment_plot_managers(fig_ax_n_store=fig_ax_n_store, plot_options_type=GridPlotOptions, plot_manager_type=GridPlotManager, plot_cycler_settings=plot_cycler_settings, mode=mode, outputs_dir=outputs_dir, metric_path_info=metric_path_info, version_dir_pattern=version_dir_pattern, final_fig_name=final_fig_name)
    return plot_managers, fig_ax_n_store

def viz_mean_correlation(dir_path: Path, mode: str) -> None:
    print(f"=========== VIZ {mode.upper()} NORMED GRAD CORR METRICS FOR REPORT ===========") 
    # Normed Per col per step (per OPTIM step)
    relative_paths_info = dict(
            relative_path= f"runs_extra-none_{mode}_Normalised-GC.parquet", 
        )
    
    NUM_PROPS_IN_CYCLE = 2
    plot_cycler_settings = dict(color=['#D95F02', '#1F78B4'], linestyle=['--', '--'], marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)

    plot_managers, fig_ax_n_store = manage_grid_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir_pattern="multijob/R*", 
        metric_path_info=relative_paths_info,
        final_fig_name="Per Layer Muon and Adam Cosine Similarity"
    ) 

    viz_plot_managers(plot_managers, outputs_dir=dir_path)

    # Normed Col reduction per step (per OPTIM step)
    relative_paths_info = dict(
            relative_path= f"runs_extra-cols_{mode}_Normalised-GC.parquet", 
    )

    plot_managers, fig_ax_n_store = manage_lone_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir_pattern="multijob/R*", 
        metric_path_info=relative_paths_info,
        final_fig_name="Muon and Adam Cosine Similarity"
    ) 

    viz_plot_managers(plot_managers, outputs_dir=dir_path)

def viz_dot_prod(dir_path: Path, mode: str) -> None:
    print(f"=========== VIZ {mode.upper()} UNNORMED GRAD CORR METRICS FOR REPORT ===========") 
    # Unnormed Per col per step (per OPTIM step)
    relative_paths_info = dict(
            relative_path= f"runs_extra-none_{mode}_Unnormalised-GC.parquet", 
        )
    
    NUM_PROPS_IN_CYCLE = 2
    plot_cycler_settings = dict(color=['#D95F02', '#1F78B4'], linestyle=['--', '--'], marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
        
    plot_managers, fig_ax_n_store = manage_grid_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir_pattern="multijob/R*", 
        metric_path_info=relative_paths_info,
        final_fig_name="Per Layer Muon and Adam Dot Product"
    ) 

    viz_plot_managers(plot_managers, outputs_dir=dir_path)

    # Unnormed Col reduction per step (per OPTIM step)
    relative_paths_info = dict(
            relative_path= f"runs_extra-cols_{mode}_Unnormalised-GC.parquet", 
        )
    
    plot_managers, fig_ax_n_store = manage_lone_metrics(
        plot_cycler_settings=plot_cycler_settings,
        mode=mode,
        outputs_dir=dir_path, 
        version_dir_pattern="multijob/R*", 
        metric_path_info=relative_paths_info,
        final_fig_name="Muon and Adam Dot Product"
    ) 

    viz_plot_managers(plot_managers, outputs_dir=dir_path)

def viz_accuracy(dir_path: Path, summary_mode: str) -> None:
    print(f"=========== VIZ {summary_mode.upper()} SUMMARY ACCURACY METRICS FOR REPORT ===========") 
    # Col reduction per epoch
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-both_{summary_mode}_Train-Accuracy.parquet", 
        ),
        dict(
            relative_path = f"runs_extra-both_{summary_mode}_Validation-Accuracy.parquet", 
        ) 
    ]

    # We run over optimisers first for train, then for val
    NUM_PROPS_IN_CYCLE = 4
    plot_cycler_settings = dict(color=['#D95F02', '#1F78B4', '#D95F02', '#1F78B4'], linestyle=['-', '-', '--', '--'], marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
    curr_fig_ax_n_store = None 
    immutable_fig_ax_store = None 
    for i, metric_file_pattern in enumerate(relative_paths_info):
        plot_managers, curr_fig_ax_n_store = manage_lone_metrics(
            fig_ax_n_store=immutable_fig_ax_store,
            plot_cycler_settings=plot_cycler_settings,
            mode=summary_mode,
            outputs_dir=dir_path, 
            version_dir_pattern="multijob/R*", 
            metric_path_info=metric_file_pattern,
            final_fig_name="Muon and Adam Train and Validation Accuracy"
        )

        if i == 0:
            immutable_fig_ax_store = curr_fig_ax_n_store

        viz_plot_managers(plot_managers, outputs_dir=dir_path)

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
    plot_cycler_settings = dict(color=['#D95F02', "#DB379C", '#1F78B4', "#4AD1C6"] * 2, linestyle=['-'] * 4 + ['--'] * 4, marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)

    curr_fig_ax_n_store = None 
    immutable_fig_ax_store = None 
    for i, metric_file_pattern in enumerate(relative_paths_info):
        plot_managers, curr_fig_ax_n_store = manage_lone_metrics(
            fig_ax_n_store=immutable_fig_ax_store,
            plot_cycler_settings=plot_cycler_settings,
            mode=summary_mode,
            outputs_dir=dir_path, 
            version_dir_pattern="multijob/R*", 
            metric_path_info=metric_file_pattern,
            final_fig_name="Subgroup Muon and Adam Train and Validation Accuracy"
        )

        if i == 0:
            immutable_fig_ax_store = curr_fig_ax_n_store

        viz_plot_managers(plot_managers, outputs_dir=dir_path)

def viz_loss(dir_path: Path, summary_mode: str) -> None:
    print(f"=========== VIZ {summary_mode.upper()} SUMMARY LOSS METRICS FOR REPORT ===========") 
    # Loss per step (NOT per optim step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-none_{summary_mode}_Train-Loss.parquet", 
        ),
        dict(
            relative_path = f"runs_extra-none_{summary_mode}_Validation-Loss.parquet", 
        ) 
    ]

    NUM_PROPS_IN_CYCLE = 4
    plot_cycler_settings = dict(color=['#D95F02', '#1F78B4', '#D95F02', '#1F78B4'], linestyle=['-', '-', '--', '--'], marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
    curr_fig_ax_n_store = None 
    immutable_fig_ax_store = None 
    
    for i, metric_file_pattern in enumerate(relative_paths_info):
        plot_managers, curr_fig_ax_n_store = manage_lone_metrics(
            fig_ax_n_store=immutable_fig_ax_store,
            plot_cycler_settings=plot_cycler_settings,
            mode=summary_mode,
            outputs_dir=dir_path, 
            version_dir_pattern="multijob/R*", 
            metric_path_info=metric_file_pattern,
            final_fig_name="Muon and Adam Train and Validation Loss"
        )

        if i == 0:
            immutable_fig_ax_store = curr_fig_ax_n_store

        viz_plot_managers(plot_managers, outputs_dir=dir_path)

def viz_over_dir(dir_path: Path) -> None:
    """ 
    Visualize non-core metrics that were created. 
    """
    SUMMARY_MODES: list[str] = ["mean"]
    
    subset_summary_modes = SUMMARY_MODES[0:] # Generate only a subset of modes especially if new modes are added in @./create_non_core_summaries.py
    for mode in subset_summary_modes: 
        viz_accuracy(dir_path=dir_path, summary_mode=mode)
        viz_loss(dir_path=dir_path, summary_mode=mode)
        viz_dot_prod(dir_path=dir_path, mode=mode)
        viz_mean_correlation(dir_path=dir_path, mode=mode)

if __name__ == "__main__":
    viz_over_dir(cls_imb_s_outputs_dir)