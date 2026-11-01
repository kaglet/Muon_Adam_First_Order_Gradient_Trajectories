from dataclasses import dataclass
import math
from typing import Any, Tuple, override

from abc import ABC, abstractmethod

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import gc
import matplotlib as mpl
import matplotlib.ticker as ticker

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pandas import DataFrame

@dataclass
class PlotOptions():
    base_size: float 
    use_cols: bool 
    plot_cycler: dict[str, list] | None
    train_split: str 
    y_label: str 
    x_label: str 
    optim_name: str 
    mode: str 
    version: str 
    final_fig_name: str 
    plot_name: str

    def __post_init__(self) -> None:
        self.names: dict[str, Any] = dict(plot_type_name=self.plot_name, train_split=self.train_split, x_label = self.x_label, y_label = self.y_label, optim_name=self.optim_name, mode=self.mode, final_fig_name=self.final_fig_name, version=self.version)

@dataclass
class LonePlotOptions(PlotOptions):
    pass 

@dataclass(init=False)
class GridPlotOptions(PlotOptions):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        kwds["grid_col_num"] = 5 # This is cheating but since it is a constant it is fine
        assert type(kwds["grid_col_num"]) is int, "`grid_col_num` must be set and have an integer number setting"
        self.grid_col_num = kwds["grid_col_num"]

class PlotManager(ABC):
    def __init__(self, metric_df: pd.DataFrame, plt_options: PlotOptions) -> None:
        self._metric_df = metric_df
        self._plot_options = plt_options
        self._fig_ax_n = self._create_fig_ax_n()

    @abstractmethod
    def _create_fig_ax_n(self) -> Tuple[Figure, Any]:
        pass 

    @abstractmethod
    def viz_cols_in_ax_n(self) -> Tuple[Figure, Any]:
        pass 

    @property 
    def plot_option_names(self):
        return self._plot_options.names
    
    @property 
    def fig_ax_n(self):
        return self._fig_ax_n

    @fig_ax_n.setter
    def fig_ax_n(self, fig_ax_n: Tuple[Figure, Any]) -> None:
        self._fig_ax_n = fig_ax_n

    def overwrite_fig_ax_n(self, fig_ax_n: Tuple[Figure, Any]) -> bool:
        self.fig_ax_n = fig_ax_n
        return True

class LonePlotManager(PlotManager):
    def __init__(self, metric_df: pd.DataFrame, plt_options: LonePlotOptions) -> None:
        super().__init__(metric_df, plt_options)

    def _create_fig_ax_n(self) -> Tuple[Figure, Any]:
        fig = Figure(figsize = ((self._plot_options.base_size + 7), (self._plot_options.base_size + 7)))
        ax = fig.subplots(1, 1, sharex=True, sharey=True)
        NUM_PROPS_IN_CYCLE = 4
        if self._plot_options.plot_cycler is not None and type(self._plot_options.plot_cycler) is dict:
            ax.set_prop_cycle(**self._plot_options.plot_cycler)
        ax.tick_params(axis='both', labelsize='large')

        figax = fig, ax
        return figax

    def viz_cols_in_ax_n(self) -> Tuple[Figure, Axes]:
        """
        For a metric object instance, visualize all col of `values` with its associated string in `header`.
        Internal output is figure with 1 plot for all column , labelled by col name in Index.
        Optionally save as compressed numpy file with flags per viz (Recommended).
        Optionally save figure (Recommended) per viz. 
        """
        fig, ax = self.fig_ax_n
        title = self._plot_options.names["mode"]  + " " + self._plot_options.names["final_fig_name"]
        fig.suptitle(f'{title}', fontsize="xx-large", y=0.92)

        for col in self._metric_df.columns:
            line, = ax.plot(list(self._metric_df.index.to_numpy(dtype=np.int32)), list(self._metric_df.loc[:, col].values))

            og_col = col
            col = f'{self._plot_options.names["optim_name"]} {self._plot_options.names["train_split"]}'
            if self._plot_options.use_cols:
                col = col + " " + og_col
            line.set_label(col.title())
            ax.legend(fontsize='medium')
        
        ax.grid(axis='both', color='gray', linestyle='--', linewidth=1, alpha=0.65, which="major")

        ax.set_ylabel(self._plot_options.names["y_label"].title(), fontsize='x-large')
        ax.set_xlabel(self._plot_options.names["x_label"].title(), fontsize='x-large')

        return fig, ax

    def overwrite_fig_ax_n(self, fig_ax_n: Tuple[Figure, Any]) -> bool:
        return super().overwrite_fig_ax_n(fig_ax_n)
    
class GridPlotsmanager(PlotManager):
    def __init__(self, metric_df: DataFrame, plt_options: GridPlotOptions) -> None:
        super().__init__(metric_df, plt_options)
        self._plot_options = plt_options

    def _create_fig_ax_n(self) -> Tuple[Figure, Any]:
        self._total_cols = len(self._metric_df.columns)
        self._grid_rows_num = math.ceil(self._total_cols / self._plot_options.grid_col_num)
        fig = Figure(figsize=(((self._plot_options.base_size + 6) * self._plot_options.grid_col_num, (self._plot_options.base_size - 2) * self._grid_rows_num) ))
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        axs = fig.subplots(
            ncols=self._plot_options.grid_col_num,
            nrows=self._grid_rows_num,
            sharex=True,
            sharey=True,
            squeeze=False
            # dpi=self._plot_options.dpi,
        )
        fig.subplots_adjust(top=0.90)

        for grid_row in range(self._grid_rows_num):
            for grid_col in range(self._plot_options.grid_col_num):
                ax = axs[grid_row, grid_col]
                if self._plot_options.plot_cycler is not None and type(self._plot_options.plot_cycler) is dict:
                    ax.set_prop_cycle(**self._plot_options.plot_cycler)
                    
        figaxes = fig, axs
        return figaxes

    def viz_cols_in_ax_n(self) -> Tuple[Figure, Any]:
        # figsize is (w,h) as spec
        fig, axs = self.fig_ax_n
        title = self._plot_options.names["mode"]  + " " + self._plot_options.names["final_fig_name"] + " " + self._plot_options.names["x_label"]

        fig.suptitle(f'{title}', fontsize="xx-large", y=0.92)

        cols_limit_counter = 0
        for grid_col in range(self._plot_options.grid_col_num):
            for grid_row in range(self._grid_rows_num):
                if cols_limit_counter >= self._total_cols:
                    break 

                col = self._metric_df.columns[cols_limit_counter]
                index_vals = list(self._metric_df.index.to_numpy(dtype=np.int32))
                col_vals = list(self._metric_df.loc[:, col].values)
                cols_limit_counter += 1

                ax = axs[grid_row, grid_col]
                
                ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=True, labelsize='x-small')

                ax.set_ylabel(self._plot_options.names["y_label"].title(), fontsize='medium') 
                ax.set_xlabel(self._plot_options.names["x_label"].title(), fontsize='medium')           
                ax.set_title(f"{col.title()}", fontsize='large')

                line, = ax.plot(index_vals, col_vals, label=f'{self._plot_options.names["optim_name"]}')

                ax.grid(axis='both', color='gray', linestyle='--', linewidth=1, alpha=0.65, which="major")
                ax.legend(fontsize='x-small')

        return fig, axs
            
    def overwrite_fig_ax_n(self, fig_ax_n: Tuple[Figure, Any]) -> bool:
        fig, axs = fig_ax_n
        assert type(axs) == type(self.fig_ax_n[1]), f"Type of axs to overwrite current must be {type(self.fig_ax_n[1])} not {type(axs)}."
        rows_equal = len(self.fig_ax_n[1]) == len(axs)
        cols_equal = len(self.fig_ax_n[1][0]) == len(axs[0])
        # TODO: If uneven grid would need to expand this to be a check over all dims (easier if np.ndarray)
        if rows_equal and cols_equal:
            self._fig_ax_n = (fig, axs)
        else:
            raise ValueError()
        
        return True

if __name__ == "__main__":
    mpl.rcParams['font.size'] = 150

    metric_df = pd.DataFrame(
        {
            "col1": np.random.random(100), 
            "col2": np.random.random(100),
            "col3": np.random.random(100),
            "col4": np.random.random(100),
            "col5": np.random.random(100),
            "col6": np.random.random(100),            
            "col7": np.random.random(100), 
            "col8": np.random.random(100),
            "col9": np.random.random(100),
            "col10": np.random.random(100),
            "col11": np.random.random(100),
            "col12": np.random.random(100),            
            "col13": np.random.random(100), 
            "col14": np.random.random(100),
            "col15": np.random.random(100),
            "col16": np.random.random(100),
            "col17": np.random.random(100),
            "col18": np.random.random(100),
        }
    )
    # viz_each_col_unsized(metric_df=metric_df, metadata_df=metric_df)
    # viz_cols_in_grid(metric_df=metric_df)